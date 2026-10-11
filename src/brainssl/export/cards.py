"""Model cards for the Hugging Face Hub, with numbers read from the report."""

from __future__ import annotations

from pathlib import Path


def _fmt(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def segmentation_card(results: dict) -> str:
    runs = {r["tag"]: r for r in results.get("smri", {}).get("runs", [])}
    full = runs.get("ft_ours_100pct", {})
    ten = runs.get("ft_ours_10pct", {})
    base_ten = runs.get("ft_unet_scratch_10pct", {})
    return f"""---
license: cc-by-sa-4.0
library_name: onnx
tags:
- medical-imaging
- brain-tumour-segmentation
- mri
- 3d-unet
- self-supervised-learning
datasets:
- medical-segmentation-decathlon
pipeline_tag: image-segmentation
---

# brainssl: bilateral U-Net for brain tumour segmentation

A residual 3D U-Net that segments glioma sub-regions in four-contrast brain MRI
(FLAIR, T1, T1 with gadolinium, T2). Each encoder level can read the features of
its mirror image across the brain's mid-sagittal plane, and the network was
pretrained without labels by reconstructing hidden tissue from its
contralateral twin, then fine-tuned on annotated scans.

Code, training pipeline and the full evaluation:
<https://github.com/mahdisabetkish/brainssl>. Interactive results:
<https://mahdisabetkish.github.io/brainssl/>.

## Files

| File | What it is |
|---|---|
| `segmentation.onnx` | one sliding window: input `image` (N, 4, 160, 128, 128), output `logits` (N, 3, 160, 128, 128) |
| `segmentation.json` | everything around the window: crop, normalisation, window placement, blending, thresholds |
| `model.safetensors`, `config.json` | PyTorch weights for `brainssl.smri.model.UNet3D` |

The three output channels are whole tumour, tumour core and enhancing tumour,
each through its own sigmoid; the label map is 1 oedema, 2 non-enhancing core,
3 enhancing tumour.

## Use

Without Python, with the Rust binary from the repository:

```bash
brainssl-infer --ort-lib /path/to/libonnxruntime.so segment \\
    --model . --input scan.nii.gz --output scan_seg.nii.gz
```

From Python, the `brainssl` package reproduces the same pipeline (crop about the
midline, z-score per contrast, sliding windows with Gaussian blending, mirror
averaging).

## Training data

Medical Segmentation Decathlon Task01 (BraTS 2016 and 2017 scans, CC BY-SA 4.0):
pretraining on 576 scans (310 annotated scans with their labels ignored, plus
the 266 released without labels), fine-tuning on the 310 annotated training
scans of a fixed split. Validation and test scans were never used for training.

## Evaluation

Held-out test split of 96 annotated scans, Dice (higher is better):

| Training labels | Model | WT | TC | ET |
|---|---|---|---|---|
| 100% (310 scans) | this model | {_fmt(full.get("dice_WT"))} | {_fmt(full.get("dice_TC"))} | {_fmt(full.get("dice_ET"))} |
| 10% (31 scans) | same method | {_fmt(ten.get("dice_WT"))} | {_fmt(ten.get("dice_TC"))} | {_fmt(ten.get("dice_ET"))} |
| 10% (31 scans) | U-Net from scratch | {_fmt(base_ten.get("dice_WT"))} | {_fmt(base_ten.get("dice_TC"))} | {_fmt(base_ten.get("dice_ET"))} |

## Limitations

Trained on skull-stripped scans registered to the SRI24 template at 1 mm, as
released by the BraTS organisers; other preprocessing will need the same steps
first. The mid-sagittal plane is estimated per scan, but strong midline shift
from a large tumour weakens the left-right correspondence the model relies on.
This is a research model and not a medical device.
"""


def connectome_card(results: dict) -> str:
    rows = {(r["model"], r["atlas"]): r for r in results.get("fmri_study", {}).get("kfold", {}).get("rows", [])}
    ours = rows.get(("vicinal", "ensemble"), {})
    tang = rows.get(("tangent", "ensemble"), {})

    def m(r, k):
        return _fmt(r.get(k, {}).get("mean")) if r else "n/a"

    return f"""---
license: cc-by-nc-sa-4.0
library_name: brainssl
tags:
- fmri
- connectomics
- functional-connectivity
- abide
- riemannian-geometry
datasets:
- abide
pipeline_tag: tabular-classification
---

# brainssl: noise-aware connectome classifier (ABIDE I)

Linear classifiers in the log-Euclidean embedding of resting-state functional
connectivity, one per atlas, trained on each scan's own estimation noise: the
connectome is re-estimated on six windows of the scan and the classifier is fit
to the full scan and its windows together. Predictions average over the full
scan, its windows and the five atlases.

Code and the full evaluation: <https://github.com/mahdisabetkish/brainssl>.

## Files

`connectome.json` describes the feature pipeline and, for each atlas, the
intercept, ROI labels and centroids; `weights_<atlas>.bin` holds the feature
means, standard deviations and weights as little-endian float32, in that order.

## Evaluation

ABIDE I preprocessed (C-PAC, band-pass filtered, no global signal regression),
871 subjects after quality control, 5 repeats of 10-fold cross-validation
stratified by site and diagnosis:

| Model | AUC | Accuracy | Balanced accuracy |
|---|---|---|---|
| this model (five-atlas ensemble) | {m(ours, "auc")} | {m(ours, "accuracy")} | {m(ours, "balanced_accuracy")} |
| tangent space + logistic regression (five-atlas ensemble) | {m(tang, "auc")} | {m(tang, "accuracy")} | {m(tang, "balanced_accuracy")} |

## Intended use and limitations

A research model for studying connectome classification methods. The labels
come from a pooled multi-site dataset with site-dependent proportions of autistic
participants, and accuracy around 0.7 is far from anything usable for
diagnosis. It must not be used to make or support decisions about any person.
ABIDE data are shared for non-commercial research, and the weights carry the same
restriction.
"""


def write_cards(segmentation_dir: Path, connectome_dir: Path, results: dict) -> None:
    (segmentation_dir / "README.md").write_text(segmentation_card(results), encoding="utf-8")
    (connectome_dir / "README.md").write_text(connectome_card(results), encoding="utf-8")
