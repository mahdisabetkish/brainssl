"""What the deployed connectome classifier looks at, and whether that can be trusted.

The classifier is linear in standardised log-connectivity features, so its
explanation is exact and needs no surrogate: the logit of a subject is the sum
over features of w_f * z_f, and ``w_f * z_f`` is that feature's contribution.
Three things are reported.

* Where the weight sits. Features are entries of the matrix logarithm of the
  correlation matrix, so entry (i, j) is a coupling between regions i and j seen
  through the log, not a single edge of the correlation matrix. Single entries
  are therefore not read one by one; the report sums the weight over blocks of
  regions (lobe by lobe for the AAL atlas) and over regions (node strength).
* Whether it is stable. The classifier is refitted on bootstrap resamples of the
  subjects and the weights are compared: overlap of the strongest features, and
  rank agreement of the node strengths.
* Whether the explanation is faithful. The features an explanation names as the
  most supportive for a subject are deleted (set to the training mean), and the
  held-out AUC is compared with deleting the same number of random features, or
  of the least supportive ones. An explanation that names what the model really
  uses must hurt more than a random one.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as sps

from ..fmri.connectivity import sym_to_vec
from ..fmri.final import fit_atlas
from ..fmri.methods import ConnectomeData
from ..stats import auc_score
from .measures import block_sums, top_k_overlap

log = logging.getLogger(__name__)

AAL_LOBES = {2: "frontal", 3: "insula", 4: "limbic", 5: "occipital", 6: "parietal", 7: "subcortical",
             8: "temporal", 9: "cerebellum"}


def vec_to_sym(v: np.ndarray, r: int) -> np.ndarray:
    """Symmetric (r, r) matrix holding a vectorised upper triangle (diagonal included), no rescaling."""
    m = np.zeros((r, r))
    m[np.triu_indices(r)] = v
    return m + np.triu(m, 1).T


def roi_names(atlas_dir: Path, atlas: str, labels: np.ndarray) -> list[str]:
    """Human-readable name of every ROI, from the label tables distributed with the atlases."""
    def table(name: str) -> dict[int, str]:
        out = {}
        with open(atlas_dir / name, encoding="utf-8") as fh:
            for row in csv.reader(line for line in fh if not line.startswith("#")):
                if len(row) >= 2 and row[0].strip().isdigit():
                    out[int(row[0])] = row[1].strip()
        return out

    if atlas == "aal":
        names = table("aal_labels.csv")
    elif atlas == "ho":
        names = table("ho_labels.csv")
    elif atlas == "dosenbach160":
        names = table("dos160_labels.csv")
    else:                           # cc200 / cc400: name a region after the AAL label that covers most of it
        names = {}
        fname = "CC200_ROI_labels.csv" if atlas == "cc200" else "CC400_ROI_labels.csv"
        text = (atlas_dir / fname).read_text(encoding="utf-8").splitlines()[1:]
        for line in text:
            parts = line.split(",", 5)
            if parts and parts[0].strip().isdigit() and len(parts) > 4:
                cell = parts[4].strip()
                names[int(parts[0])] = cell.split('"')[1] if '"' in cell else f"ROI {parts[0].strip()}"
    return [f"{names.get(int(l), 'ROI ' + str(int(l)))}" + ("" if atlas in ("aal", "ho") else f" ({int(l)})")
            for l in labels]


def groups_for(atlas: str, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Lobe (and hemisphere) of every ROI: from the AAL code, or from the side of the midline otherwise."""
    if atlas == "aal":
        side = np.where(np.array([int(l) % 2 for l in labels]) == 1, "L", "R")
        lobe = np.array([AAL_LOBES.get(int(l) // 1000, "other") for l in labels])
        return np.array([f"{lo} {s}" if lo != "cerebellum" else "cerebellum" for lo, s in zip(lobe, side)])
    return np.where(centroids[:, 0] < -2, "left", np.where(centroids[:, 0] > 2, "right", "midline"))


def weight_maps(connectome_dir: Path, atlas_dir: Path, top: int = 12) -> dict:
    """Node strength and block structure of the shipped weights, per atlas."""
    meta = json.loads((connectome_dir / "connectome.json").read_text())
    out = {}
    for atlas, info in meta["atlases"].items():
        raw = np.fromfile(connectome_dir / info["file"], dtype="<f4")
        d, r = info["n_features"], info["n_rois"]
        w = raw[2 * d:3 * d].astype(np.float64)
        labels = np.array(info["roi_labels"])
        centroids = np.array(info["centroids_mm"])
        W = vec_to_sym(w, r)
        names = roi_names(atlas_dir, atlas, labels)
        strength = np.abs(W).sum(1) - np.abs(np.diag(W)) / 2
        order = np.argsort(-strength)[:top]
        entry = {"n_rois": r, "n_features": d,
                 "top_rois": [{"name": names[i], "strength": float(strength[i]),
                               "signed": float(W[i].sum()), "mni_mm": centroids[i].round(1).tolist()} for i in order],
                 "strength": strength.tolist(), "names": names, "centroids_mm": centroids.round(1).tolist()}
        groups = groups_for(atlas, labels, centroids)
        absblock = block_sums(np.abs(W), groups)
        signblock = block_sums(W, groups)
        entry["blocks"] = {"names": absblock["names"],
                           "mean_abs": (absblock["sum"] / np.maximum(absblock["count"], 1)).tolist(),
                           "mean_signed": (signblock["sum"] / np.maximum(signblock["count"], 1)).tolist(),
                           "count": absblock["count"].tolist()}
        out[atlas] = entry
    return out


def _bootstrap_fits(data: ConnectomeData, atlas: str, C: float, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = []
    for b in range(n_boot):
        idx = rng.integers(0, data.n, data.n)
        sub = ConnectomeData(data.labels[idx], data.sites[idx], (atlas,), {}, {atlas: data.logs[atlas][idx]},
                             {atlas: data.windows[atlas][idx]})
        weights.append(fit_atlas(sub, atlas, C)["weight"].astype(np.float64))
        if (b + 1) % 10 == 0:
            log.info("%s bootstrap %d/%d", atlas, b + 1, n_boot)
    return np.stack(weights)


def stability(data: ConnectomeData, atlas: str, C: float, w_full: np.ndarray, n_roi: int,
              n_boot: int = 30, top_k: int = 100, seed: int = 0) -> dict:
    """Agreement of the weights of bootstrap refits with each other and with the full fit."""
    ws = _bootstrap_fits(data, atlas, C, n_boot, seed)
    strength = np.stack([np.abs(vec_to_sym(w, n_roi)).sum(1) for w in ws])
    strength_full = np.abs(vec_to_sym(w_full, n_roi)).sum(1)
    pairs = [(i, j) for i in range(n_boot) for j in range(i + 1, n_boot)]
    rng = np.random.default_rng(seed + 1)
    chosen = [pairs[k] for k in rng.choice(len(pairs), size=min(100, len(pairs)), replace=False)]
    top_full = np.argsort(-np.abs(w_full))[:top_k]
    sign_agree = (np.sign(ws[:, top_full]) == np.sign(w_full[top_full])).mean(0)
    return {"bootstraps": n_boot, "top_k": top_k,
            "top_k_jaccard_pairs": float(np.mean([top_k_overlap(ws[i], ws[j], top_k) for i, j in chosen])),
            "top_k_jaccard_with_full": float(np.mean([top_k_overlap(w, w_full, top_k) for w in ws])),
            "weight_rank_correlation": float(np.mean([sps.spearmanr(ws[i], ws[j])[0] for i, j in chosen[:30]])),
            "node_strength_rank_correlation": float(np.mean([sps.spearmanr(strength[i], strength[j])[0]
                                                             for i, j in chosen])),
            "node_strength_vs_full": float(np.mean([sps.spearmanr(s, strength_full)[0] for s in strength])),
            "top_features_sign_stable_90": float((sign_agree >= 0.9).mean()),
            "top_features_sign_agreement": float(sign_agree.mean())}


def deletion_test(data: ConnectomeData, atlas: str, C: float, repeats: int = 5,
                  fractions: tuple[float, ...] = (0.01, 0.05, 0.2), seed: int = 0) -> dict:
    """Held-out AUC after deleting the features an explanation names, against random and least-supportive ones."""
    from sklearn.model_selection import StratifiedShuffleSplit
    rng = np.random.default_rng(seed)
    full = sym_to_vec(data.logs[atlas])
    results = {f: {"supportive": [], "random": [], "least_supportive": []} for f in fractions}
    base = []
    strat = np.char.add(data.sites.astype(str), data.labels.astype(str))
    names, counts = np.unique(strat, return_counts=True)
    strat = np.where(np.isin(strat, names[counts >= 2]), strat, "rest")
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=0.5, random_state=seed)
    for tr, te in splitter.split(full, strat):
        sub = ConnectomeData(data.labels[tr], data.sites[tr], (atlas,), {}, {atlas: data.logs[atlas][tr]},
                             {atlas: data.windows[atlas][tr]})
        fit = fit_atlas(sub, atlas, C)
        z = (full[te] - fit["mean"]) / fit["std"]
        w = fit["weight"].astype(np.float64)
        y = data.labels[te]
        contrib = z * w
        score = contrib.sum(1)
        base.append(auc_score(y, score))
        direction = np.where(score >= 0, 1.0, -1.0)[:, None]
        support = contrib * direction                      # large when a feature pushes towards the predicted class
        for frac in fractions:
            k = max(1, int(round(frac * z.shape[1])))
            for name, picks in (("supportive", np.argsort(-support, 1)[:, :k]),
                                ("least_supportive", np.argsort(support, 1)[:, :k]),
                                ("random", np.stack([rng.choice(z.shape[1], k, replace=False)
                                                     for _ in range(len(te))]))):
                zz = z.copy()
                np.put_along_axis(zz, picks, 0.0, axis=1)
                results[frac][name].append(auc_score(y, zz @ w))
    return {"baseline_auc": float(np.mean(base)),
            "fractions": {str(f): {k: float(np.mean(v)) for k, v in r.items()} for f, r in results.items()}}


def interpret(raw: Path, processed: Path, connectome_dir: Path, choices: dict, out: Path | None = None,
              stability_atlases: tuple[str, ...] = ("aal", "ho", "dosenbach160", "cc200"),
              deletion_atlases: tuple[str, ...] = ("aal",), n_boot: int = 30) -> dict:
    atlas_dir = raw / "abide" / "atlases"
    report = {"weights": weight_maps(connectome_dir, atlas_dir), "stability": {}, "deletion": {}}
    meta = json.loads((connectome_dir / "connectome.json").read_text())
    for atlas in sorted(set(stability_atlases) | set(deletion_atlases)):
        data = ConnectomeData.load(raw, processed, (atlas,), windows=True)
        C = choices["vicinal_C"][atlas]
        info = meta["atlases"][atlas]
        d = info["n_features"]
        w_full = np.fromfile(connectome_dir / info["file"], dtype="<f4")[2 * d:3 * d].astype(np.float64)
        if atlas in stability_atlases:
            report["stability"][atlas] = stability(data, atlas, C, w_full, info["n_rois"], n_boot=n_boot)
        if atlas in deletion_atlases:
            report["deletion"][atlas] = deletion_test(data, atlas, C)
        del data
    if out is not None:
        from ..utils import write_json
        write_json(out, report)
    return report
