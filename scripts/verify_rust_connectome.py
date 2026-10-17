"""Check that the Rust binary scores scans exactly like the Python export.

For a few subjects, computes the probability in Python from the exported
connectome model (same feature pipeline as ``brainssl.fmri.final``) and with
``brainssl-infer classify``, and compares them.

    python scripts/verify_rust_connectome.py --model exports/connectome --atlas cc200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brainssl.fmri.abide import load_cache, load_phenotypes
from brainssl.fmri.connectivity import shrunk_correlation, spd_log, sym_to_vec
from brainssl.fmri.windows import window_starts


def python_probability(ts: np.ndarray, model_dir: Path, atlas: str) -> float:
    meta = json.loads((model_dir / "connectome.json").read_text())
    spec = meta["atlases"][atlas]
    params = np.fromfile(model_dir / spec["file"], dtype="<f4").astype(np.float64)
    d = spec["n_features"]
    mean, std, w = params[:d], params[d:2 * d], params[2 * d:]
    def score(x: np.ndarray) -> float:
        z = (sym_to_vec(spd_log(shrunk_correlation(x))) - mean) / std
        return float(1 / (1 + np.exp(-(z @ w + spec["bias"]))))
    starts, width = window_starts(len(ts), meta["windows"]["count"], meta["windows"]["fraction"])
    p_full = score(ts)
    p_win = np.mean([score(ts[s:s + width]) for s in starts])
    return float(0.5 * (p_full + p_win))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--atlas", default="cc200")
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--binary", type=Path, default=Path("rust/target/release/brainssl-infer"))
    parser.add_argument("--ort-lib", type=Path, default=None,
                        help="ONNX Runtime shared library; defaults to the one bundled with onnxruntime")
    args = parser.parse_args()
    lib = args.ort_lib
    if lib is None:
        capi = Path(ort.__file__).parent / "capi"
        found = sorted(capi.glob("onnxruntime.dll")) + sorted(capi.glob("libonnxruntime.so*")) \
            + sorted(capi.glob("libonnxruntime*.dylib"))
        if not found:
            sys.exit("could not find the ONNX Runtime library; pass --ort-lib")
        lib = found[0]
    binary = args.binary.with_suffix(".exe") if os.name == "nt" else args.binary
    pheno = load_phenotypes(args.raw / "abide")
    ts_set = load_cache(args.processed / "abide", args.atlas).subset(pheno["FILE_ID"].tolist())
    worst = 0.0
    for i in range(args.subjects):
        ts = ts_set[i].astype(np.float64)
        path = args.model / f"_parity_{i}.1D"
        np.savetxt(path, ts, fmt="%.9g")
        run = subprocess.run([str(binary), "--ort-lib", str(lib), "classify", "--model", str(args.model),
                              "--timeseries", str(path), "--atlas", args.atlas], capture_output=True, text=True)
        path.unlink()
        if run.returncode != 0:
            sys.exit(f"rust binary failed with exit code {run.returncode}:\n{run.stderr}")
        p_rust = json.loads(run.stdout)["probability"]
        p_py = python_probability(ts, args.model, args.atlas)
        worst = max(worst, abs(p_rust - p_py))
        print(f"{pheno['FILE_ID'].iloc[i]}: python {p_py:.6f}  rust {p_rust:.6f}")
    print(f"largest difference {worst:.2e}")
    if worst > 1e-4:
        sys.exit("parity check failed")
    print("parity check passed")


if __name__ == "__main__":
    main()
