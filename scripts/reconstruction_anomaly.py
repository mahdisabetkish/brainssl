"""Does contralateral reconstruction error point at tumours, without labels?

Runs a pretrained network on validation scans under several random
contralateral masks, averages the squared reconstruction error of every hidden
voxel, and scores that error as a voxel-wise tumour detector against the
annotation (used for scoring only). Validation scans were not used for
pretraining.

    python scripts/reconstruction_anomaly.py --run runs/smri/pretrain_bilateral_contralateral
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from brainssl.smri.masking import contralateral_mask, random_mask
from brainssl.smri.model import UNet3D, UNetConfig
from brainssl.smri.preprocess import load_case
from brainssl.utils import write_json


@torch.no_grad()
def error_map(model: UNet3D, image: torch.Tensor, masking: str, draws: int, gen: torch.Generator) -> np.ndarray:
    """Mean squared reconstruction error per voxel, over draws in which it was hidden."""
    c, X, Y, Z = image.shape
    win = (X, 128, 128)
    total = torch.zeros((X, Y, Z), device=image.device)
    count = torch.zeros((X, Y, Z), device=image.device)
    starts_y = sorted({0, max(0, Y - win[1]), *range(0, max(1, Y - win[1] + 1), 64)})
    starts_z = sorted({0, max(0, Z - win[2]), *range(0, max(1, Z - win[2] + 1), 64)})
    for _ in range(draws):
        for y0 in starts_y:
            for z0 in starts_z:
                tile = image[:, :, y0:y0 + win[1], z0:z0 + win[2]][None]
                shape = tuple(tile.shape[2:])
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
                spatial = (contralateral_mask(1, shape, both=0.0, device=image.device) if masking == "contralateral"
                           else random_mask(1, shape, ratio=0.5, device=image.device))
                hidden = spatial[0].bool()
                with torch.autocast("cuda", dtype=torch.float16):
                    recon = model.forward_extra(tile * (1 - spatial[:, None]), "recon")[0].float()
                err = ((recon - tile[0]) ** 2).mean(0)
                total[:, y0:y0 + win[1], z0:z0 + win[2]] += err * hidden
                count[:, y0:y0 + win[1], z0:z0 + win[2]] += hidden
    return (total / count.clamp_min(1)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--masking", default=None, help="defaults to the run's own masking")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    masking = args.masking or cfg["masking"]
    model = UNet3D(UNetConfig(bilateral=cfg["bilateral"], extra_heads={"recon": 4}))
    model.load_state_dict(torch.load(args.run / "final.pt", map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    index = pd.read_csv(cfg["index"]).set_index("case_id")
    cases = json.loads(Path(cfg["splits"]).read_text())["val"][: args.cases]
    gen = torch.Generator().manual_seed(0)
    rows = []
    for case_id in cases:
        image, label, _ = load_case(index.loc[case_id], raw_root=cfg.get("raw", "data/raw"))
        err = error_map(model, torch.from_numpy(image).cuda(), masking, args.draws, gen)
        brain = (image != 0).any(axis=0)
        tumour = (label > 0)[brain]
        score = err[brain]
        rows.append({"case_id": case_id, "auroc": float(roc_auc_score(tumour, score)),
                     "error_tumour": float(score[tumour].mean()), "error_healthy": float(score[~tumour].mean())})
        print(f"{case_id}: voxel AUROC {rows[-1]['auroc']:.3f}  error tumour {rows[-1]['error_tumour']:.3f} "
              f"healthy {rows[-1]['error_healthy']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    summary = {"run": args.run.name, "masking": masking, "cases": len(df), "draws": args.draws,
               "voxel_auroc_mean": float(df.auroc.mean()), "voxel_auroc_median": float(df.auroc.median()),
               "error_ratio_tumour_to_healthy": float((df.error_tumour / df.error_healthy).median())}
    out = args.out or args.run / "reconstruction_anomaly.json"
    write_json(out, {"summary": summary, "cases": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
