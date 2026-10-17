"""Trust audit of the tumour segmentation models.

For each trained run the audit scores the held-out test scans once more, this
time keeping what a user would want to know before relying on a prediction:

* uncertainty: per-scan entropy of the probabilities and disagreement between
  the plain and the mirrored pass, and whether either ranks the scans by how
  badly they were segmented (Spearman correlation and a risk-coverage curve);
* calibration: do voxel probabilities mean what they say, in the neighbourhood
  of the tumour where the decision is hard;
* volume agreement: predicted against reference tumour volume per region;
* failure analysis: Dice by tumour size, and the worst scans with a tag for
  what went wrong;
* robustness: Dice under noise, a smooth bias field, blur and the loss of each
  contrast, on a fixed random subset of the scans and identical perturbations
  for every model;
* reliance on the contralateral half (bilateral models only): Dice when the
  exchange blocks see the scan's own features instead of its mirror image, how
  large the blocks' correction is inside and outside the tumour, and how well
  the left-right asymmetry signal alone finds the tumour.

Nothing here changes a model; every number comes from forward passes (and one
backward pass for the saliency maps). The analysis plan is in docs/trust.md.
"""

from __future__ import annotations

import json
import logging
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from ..smri.evaluate import load_model
from ..smri.infer import enforce_hierarchy, sliding_window
from ..smri.metrics import region_metrics
from ..smri.model import BilateralExchange
from ..smri.preprocess import load_case, regions_from_labels
from ..stats import auc_score, paired_case_test
from ..utils import write_json
from .measures import agreement, rank_correlation, risk_coverage

log = logging.getLogger(__name__)

REGIONS = ("WT", "TC", "ET")
CONTRASTS = ("FLAIR", "T1", "T1Gd", "T2")
BINS = 15
PERTURBATIONS = ("noise 0.2", "noise 0.5", "bias field", "blur", "no FLAIR", "no T1", "no T1Gd", "no T2")


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def perturb(image: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    """A perturbed copy of a normalised image (4, X, Y, Z); background stays zero."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    brain = (image != 0).any(0, keepdim=True)
    out = image.clone()
    if name.startswith("noise"):
        sigma = float(name.split()[1])
        out = out + sigma * torch.randn(image.shape, generator=g).to(image.device) * brain
    elif name == "bias field":
        coarse = torch.randn(1, 1, 4, 4, 4, generator=g).to(image.device)
        field = F.interpolate(coarse, size=image.shape[1:], mode="trilinear", align_corners=True)[0]
        out = out * torch.exp(0.4 * field)
    elif name == "blur":
        t = torch.arange(-3, 4, dtype=torch.float32, device=image.device)
        k = torch.exp(-0.5 * (t / 1.5) ** 2)
        k = (k / k.sum())
        x = out[:, None]
        for dim in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[2 + dim] = 7
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (2 - dim):2 * (2 - dim) + 2] = [3, 3]
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
        out = x[:, 0]
    elif name.startswith("no "):
        out[CONTRASTS.index(name[3:])] = 0
    else:
        raise ValueError(name)
    return out * brain if name != "bias field" else out


def _seed(case_id: str, name: str) -> int:
    return zlib.crc32(f"{case_id}|{name}".encode()) % (2 ** 31)


@contextmanager
def mirror_blind():
    """Let every exchange block see the scan's own features as the 'mirror image'.

    The block then receives (f, f, 0): it still runs, but carries no
    information about the other hemisphere. This input was never seen in
    training, so the drop in Dice is an upper bound on what the contralateral
    channel contributes.
    """
    original = BilateralExchange.forward

    def forward(self, x):
        return x + self.mix(torch.cat([x, x, torch.zeros_like(x)], dim=1))

    BilateralExchange.forward = forward
    try:
        yield
    finally:
        BilateralExchange.forward = original


@contextmanager
def record_exchange():
    """Collect, per exchange block, maps of |correction|, |feature| and |f - Rf| (channel means)."""
    original = BilateralExchange.forward
    store: list[dict[str, torch.Tensor]] = []

    def forward(self, x):
        mirror = torch.flip(x, dims=(2,))
        delta = self.mix(torch.cat([x, mirror, (x - mirror).abs()], dim=1))
        store.append({"delta": delta.abs().mean(1)[0].float().cpu(), "feat": x.abs().mean(1)[0].float().cpu(),
                      "asym": (x - mirror).abs().mean(1)[0].float().cpu()})
        return x + delta

    BilateralExchange.forward = forward
    try:
        yield store
    finally:
        BilateralExchange.forward = original


def _predict(model, image: torch.Tensor, window, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        return sliding_window(model, image, window=window, mirror=True, first_view=True)


def _score(prob: torch.Tensor, regions: np.ndarray) -> dict[str, float]:
    return region_metrics(enforce_hierarchy(prob), regions, distances=False)


def _tumour_window(image: torch.Tensor, wt: np.ndarray, window: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    """Window of the model's size centred on the tumour, clamped to the volume."""
    idx = np.argwhere(wt) if wt.any() else np.argwhere((image != 0).any(0).cpu().numpy())
    centre = idx.mean(0).round().astype(int)
    out = []
    for c, w, size in zip(centre, window, image.shape[1:]):
        start = int(np.clip(c - w // 2, 0, max(0, size - w)))
        out.append(slice(start, start + min(w, size)))
    return tuple(out)


def case_stats(prob: torch.Tensor, first: torch.Tensor, regions: np.ndarray) -> tuple[dict, dict]:
    """Uncertainty summaries, volumes and the calibration histogram of one scan."""
    p = prob.float()
    p2 = (2 * p - first.float()).clamp(0, 1)
    disagreement = (first.float() - p2).abs()
    ent = _entropy(p)
    row: dict[str, float] = {}
    hist = np.zeros((3, 3, BINS))                  # region, [count, sum of p, sum of y], bin
    pred = enforce_hierarchy(prob)
    for k, name in enumerate(REGIONS):
        plausible = p[k] > 0.05
        n = int(plausible.sum())
        row[f"unc_entropy_{name}"] = float(ent[k][plausible].mean()) if n else 0.0
        row[f"unc_disagree_{name}"] = float(disagreement[k][plausible].mean()) if n else 0.0
        row[f"vol_pred_{name}"] = int(pred[k].sum())
        row[f"vol_true_{name}"] = int(regions[k].sum())
        near = pred[k].astype(bool) | regions[k].astype(bool)
        if near.any():
            box = tuple(slice(max(0, a.min() - 6), a.max() + 7) for a in np.nonzero(near))
            zone = ndimage.binary_dilation(near[box], iterations=5)
            pk = p[k].cpu().numpy()[box][zone]
            yk = regions[k][box][zone].astype(float)
            b = np.minimum((pk * BINS).astype(int), BINS - 1)
            np.add.at(hist[k, 0], b, 1)
            np.add.at(hist[k, 1], b, pk)
            np.add.at(hist[k, 2], b, yk)
    return row, {"hist": hist}


def exchange_stats(model, image: torch.Tensor, regions: np.ndarray, window, amp: bool) -> list[dict[str, float]]:
    """How much each exchange block corrects inside and outside the tumour, and how well |f - Rf| finds it."""
    box = _tumour_window(image, regions[0], window)
    tile = image[(slice(None),) + box][None]
    wt = torch.from_numpy(regions[0][box].astype(np.float32))[None, None]
    brain = (image[(slice(None),) + box] != 0).any(0).float()[None, None]
    levels = []
    with torch.no_grad(), record_exchange() as store:
        with torch.autocast(tile.device.type, dtype=torch.float16, enabled=amp):
            model(tile)
    for maps in store:
        size = maps["delta"].shape
        tum = F.adaptive_avg_pool3d(wt, size)[0, 0] >= 0.25
        br = F.adaptive_avg_pool3d(brain.cpu(), size)[0, 0] >= 0.5
        out = br & ~tum
        if tum.sum() < 4 or out.sum() < 4:
            levels.append({})
            continue
        delta, feat = maps["delta"], maps["feat"]
        sel = torch.nonzero(br.reshape(-1)).squeeze(1)
        score = maps["asym"].reshape(-1)[sel].numpy()
        truth = tum.reshape(-1)[sel].numpy()
        levels.append({"correction_in": float(delta[tum].mean() / (feat[tum].mean() + 1e-8)),
                       "correction_out": float(delta[out].mean() / (feat[out].mean() + 1e-8)),
                       "asymmetry_auc": auc_score(truth, score) if truth.any() and not truth.all() else float("nan")})
    return levels


def saliency(model, image: torch.Tensor, regions: np.ndarray, window,
             amp: bool) -> tuple[np.ndarray, np.ndarray, tuple]:
    """|gradient x input| of the whole-tumour logit summed over the segmented tumour, one window.

    Returns the map summed over contrasts, the share of the attribution mass
    that each contrast carries inside the tumour, and the window position.
    """
    box = _tumour_window(image, regions[0], window)
    tile = image[(slice(None),) + box][None].clone().requires_grad_(True)
    with torch.autocast(tile.device.type, dtype=torch.float16, enabled=amp):
        logits = model(tile)
    logits = logits[0] if isinstance(logits, (list, tuple)) else logits
    mask = (logits[0, 0].detach() > 0).float()
    if mask.sum() == 0:
        mask = torch.from_numpy(regions[0][box].astype(np.float32)).to(tile.device)
    (logits[0, 0].float() * mask).sum().backward()
    attr = (tile.grad[0] * tile[0].detach()).abs().float()
    share = (attr * mask).sum((1, 2, 3))
    share = (share / share.sum().clamp_min(1e-12)).cpu().numpy()
    return attr.sum(0).cpu().numpy().astype(np.float16), share, box


def audit_run(run: Path, split: str = "test", robust_cases: int = 24, saliency_cases: int = 24,
              layers_for: list[str] | None = None, layers_dir: Path | None = None, raw: str | None = None) -> dict:
    """The full audit of one run. Slow: about ten forward passes per robustness scan."""
    cfg = json.loads((run / "config.json").read_text())
    splits = json.loads(Path(cfg["splits"]).read_text())
    index = pd.read_csv(cfg["index"]).set_index("case_id")
    case_ids = splits[split]
    model = load_model(run)
    device = next(model.parameters()).device
    amp = cfg.get("amp", True) and device.type == "cuda"
    window = (cfg.get("lr_width", 160),) + tuple(cfg.get("patch", (128, 128)))
    raw = raw or cfg.get("raw", "data/raw")
    rng = np.random.default_rng(0)
    robust_ids = set(rng.permutation(sorted(case_ids))[:robust_cases].tolist())
    sal_ids = set(sorted(robust_ids)[:saliency_cases])
    bilateral = bool(cfg["bilateral"])
    rows, hist = [], np.zeros((3, 3, BINS))
    robust: dict[str, list[dict]] = {name: [] for name in PERTURBATIONS}
    robust["clean"], robust["mirror blind"] = [], []
    gates: list[list[dict]] = []
    shares = []
    for n, case_id in enumerate(case_ids):
        image_np, label, box = load_case(index.loc[case_id], lr_width=window[0], raw_root=raw)
        regions = regions_from_labels(label)
        image = torch.from_numpy(image_np).to(device)
        with torch.no_grad():
            prob, first = _predict(model, image, window, amp)
        scores = _score(prob, regions)
        stats, extra = case_stats(prob, first, regions)
        rows.append({"case_id": case_id, **scores, **stats})
        hist += extra["hist"]
        if layers_dir is not None and case_id in (layers_for or ()):
            layers_dir.mkdir(parents=True, exist_ok=True)
            ent = _entropy(prob[0]).cpu().numpy().astype(np.float16)
            dis = (first[0] - (2 * prob[0] - first[0]).clamp(0, 1)).abs().cpu().numpy().astype(np.float16)
            sal, share, sbox = saliency(model, image, regions, window, amp)
            full = np.zeros(ent.shape, np.float16)
            full[tuple(sbox)] = sal
            np.savez_compressed(layers_dir / f"{case_id}.npz", entropy=ent, disagreement=dis, saliency=full,
                                box=np.array([[s.start, s.stop] for s in box]))
        if case_id in robust_ids:
            robust["clean"].append({"case_id": case_id, **scores})
            for name in PERTURBATIONS:
                with torch.no_grad():
                    q, _ = _predict(model, perturb(image, name, _seed(case_id, name)), window, amp)
                robust[name].append({"case_id": case_id, **_score(q, regions)})
            if bilateral:
                with mirror_blind(), torch.no_grad():
                    q, _ = _predict(model, image, window, amp)
                robust["mirror blind"].append({"case_id": case_id, **_score(q, regions)})
                gates.append(exchange_stats(model, image, regions, window, amp))
        if case_id in sal_ids:
            _, share, _ = saliency(model, image, regions, window, amp)
            shares.append(share)
        if (n + 1) % 10 == 0:
            log.info("%s: %d/%d scans", run.name, n + 1, len(case_ids))
    return {"run": run.name, "bilateral": bilateral, "cases": rows, "histogram": hist.tolist(),
            "robust": robust, "gates": gates, "saliency_share": np.array(shares).tolist()}


def summarise_run(raw: dict) -> dict:
    """Turn the raw audit of one run into the numbers that are reported."""
    df = pd.DataFrame(raw["cases"]).set_index("case_id")
    out: dict = {"run": raw["run"], "cases": int(len(df)), "bilateral": raw["bilateral"]}
    # calibration of voxel probabilities near the tumour
    hist = np.array(raw["histogram"])
    cal = {}
    for k, name in enumerate(REGIONS):
        count, sp, sy = hist[k]
        total = count.sum()
        keep = count > 0
        ece = float((count[keep] / total * np.abs(sy[keep] / count[keep] - sp[keep] / count[keep])).sum())
        cal[name] = {"ece": ece, "voxels": int(total),
                     "bins": [{"mean_pred": float(sp[b] / count[b]), "observed": float(sy[b] / count[b]),
                               "n": int(count[b])} for b in range(BINS) if count[b] >= 200]}
    out["calibration"] = cal
    # uncertainty against error
    unc = {}
    for name in REGIONS:
        err = 1 - df[f"dice_{name}"].to_numpy()
        entry = {}
        for kind in ("entropy", "disagree"):
            u = df[f"unc_{kind}_{name}"].to_numpy()
            entry[kind] = {"spearman": rank_correlation(u, err, n_boot=500),
                           "risk_coverage": risk_coverage(err, -u, grid=10)}
        unc[name] = entry
    out["uncertainty"] = unc
    # volumes
    out["volume"] = {name: agreement(df[f"vol_pred_{name}"].to_numpy(float)[df[f"vol_true_{name}"] > 0] / 1000,
                                      df[f"vol_true_{name}"].to_numpy(float)[df[f"vol_true_{name}"] > 0] / 1000)
                     for name in REGIONS}
    out["volume_points"] = {name: {"pred": (df[f"vol_pred_{name}"] / 1000).round(2).tolist(),
                                   "true": (df[f"vol_true_{name}"] / 1000).round(2).tolist(),
                                   "case": df.index.tolist()} for name in REGIONS}
    # failure analysis
    size = pd.qcut(df["vol_true_WT"], 3, labels=["small", "medium", "large"])
    out["by_size"] = {s: {"cases": int((size == s).sum()),
                          "wt_range_ml": [float(df.loc[size == s, "vol_true_WT"].min() / 1000),
                                          float(df.loc[size == s, "vol_true_WT"].max() / 1000)],
                          **{f"dice_{n}": float(df.loc[size == s, f"dice_{n}"].mean()) for n in REGIONS}}
                      for s in ("small", "medium", "large")}
    worst = df.sort_values("dice_mean").head(8)
    tags = []
    for case_id, r in worst.iterrows():
        tag = []
        if r["vol_true_ET"] > 0 and r["vol_pred_ET"] == 0:
            tag.append("enhancing tumour missed")
        if r["vol_true_ET"] == 0 and r["vol_pred_ET"] > 0:
            tag.append("enhancing tumour predicted where none exists")
        if r["vol_true_WT"] < 5000:
            tag.append("small tumour (< 5 mL)")
        if r["vol_true_WT"] > 0 and r["vol_pred_WT"] < 0.7 * r["vol_true_WT"]:
            tag.append("whole tumour under-segmented")
        if r["vol_true_WT"] > 0 and r["vol_pred_WT"] > 1.3 * r["vol_true_WT"]:
            tag.append("whole tumour over-segmented")
        tags.append({"case": case_id, "dice_mean": float(r["dice_mean"]),
                     "dice": {n: float(r[f"dice_{n}"]) for n in REGIONS},
                     "tags": tag or ["no single dominant cause"]})
    out["worst_cases"] = tags
    out["et_absent"] = {"cases": int((df["vol_true_ET"] == 0).sum()),
                        "false_alarm": int(((df["vol_true_ET"] == 0) & (df["vol_pred_ET"] > 0)).sum())}
    # robustness (the scans the perturbations were applied to)
    clean = pd.DataFrame(raw["robust"]["clean"]).set_index("case_id")
    rob = {"scans": int(len(clean)), "clean_dice": {n: float(clean[f"dice_{n}"].mean()) for n in REGIONS},
           "clean_mean": float(clean["dice_mean"].mean()), "perturbations": {}}
    for name in list(PERTURBATIONS) + ["mirror blind"]:
        if not raw["robust"].get(name):
            continue
        p = pd.DataFrame(raw["robust"][name]).set_index("case_id").loc[clean.index]
        rob["perturbations"][name] = {
            "dice": {n: float(p[f"dice_{n}"].mean()) for n in REGIONS}, "mean": float(p["dice_mean"].mean()),
            "drop": {n: float((clean[f"dice_{n}"] - p[f"dice_{n}"]).mean()) for n in REGIONS},
            "drop_mean": float((clean["dice_mean"] - p["dice_mean"]).mean()),
            "per_case_drop": (clean["dice_mean"] - p["dice_mean"]).round(4).to_dict()}
    out["robustness"] = rob
    if raw["gates"]:
        levels = []
        for lv in range(len(raw["gates"][0])):
            vals = [g[lv] for g in raw["gates"] if g[lv]]
            if not vals:
                levels.append({})
                continue
            entry = {k: float(np.nanmean([v[k] for v in vals]))
                     for k in ("correction_in", "correction_out", "asymmetry_auc")}
            levels.append(entry | {"scans": len(vals)})
        out["exchange"] = levels
    if raw["saliency_share"]:
        out["saliency_share"] = dict(zip(CONTRASTS, np.mean(raw["saliency_share"], axis=0).round(4).tolist()))
    return out


def compare_robustness(a: dict, b: dict) -> dict:
    """Paired per-scan comparison of the Dice drop of two runs (positive: run a loses less)."""
    out = {}
    for name, pa in a["robustness"]["perturbations"].items():
        pb = b["robustness"]["perturbations"].get(name)
        if pb is None:
            continue
        ids = sorted(set(pa["per_case_drop"]) & set(pb["per_case_drop"]))
        da = np.array([pa["per_case_drop"][i] for i in ids])
        db = np.array([pb["per_case_drop"][i] for i in ids])
        out[name] = paired_case_test(db, da)          # mean_diff > 0: b dropped more than a
    return out


def audit(runs: dict[str, Path], out: Path, layers_for: list[str] | None = None, layers_root: Path | None = None,
          robust_cases: int = 24, raw: str | None = None, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Audit several runs and write one JSON; raw per-run results are cached next to it.

    ``pairs`` lists (proposed, reference) labels whose robustness is compared case by case.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label, run in runs.items():
        cache = out.parent / f"trust_smri_raw_{label}.json"
        if cache.exists():
            raw_result = json.loads(cache.read_text())
        else:
            raw_result = audit_run(run, robust_cases=robust_cases, saliency_cases=robust_cases, layers_for=layers_for,
                                   layers_dir=(layers_root / label) if layers_root else None, raw=raw)
            write_json(cache, raw_result)
        summaries[label] = summarise_run(raw_result)
        log.info("audited %s", label)
    report = {"runs": summaries,
              "robustness_comparisons": {f"{a}|{b}": compare_robustness(summaries[a], summaries[b])
                                         for a, b in (pairs or [])}}
    write_json(out, report)
    return report
