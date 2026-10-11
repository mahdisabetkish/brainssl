# brainssl

Two studies on public neuroimaging data that ask the same question from
different directions: how much can a model gain from structure in the data
that the standard baseline ignores, when annotations are scarce or noisy?

**Structural MRI.** Brain tumour segmentation on four-contrast MRI from the
Medical Segmentation Decathlon. A 3D U-Net gets one extra block per encoder
level, which lets every voxel compare itself with its mirror image across the
brain's midline, and is pretrained without labels by reconstructing hidden
tissue from its contralateral twin. It is compared with the same U-Net trained
from scratch, at 5, 10, 25 and 100% of the annotated scans.

**Resting-state fMRI.** Autism versus control classification from functional
connectomes in ABIDE I. The reference pipeline (tangent-space connectivity and a
penalised linear classifier) treats each subject's connectome as an exact
point; here each scan's own estimation noise, measured by re-estimating the
connectome on windows of the scan, becomes the training distribution. It is
compared with that reference pipeline and with three published deep models.

## Where to start

- New to the imaging side: the primers on [brain tumour MRI](primer/mri.md) and
  [resting-state fMRI](primer/fmri.md) explain the data, the preprocessing and
  the vocabulary for readers who come from machine learning.
- The methods, with their derivations: [bilateral U-Net and contralateral
  masking](methods/smri.md), [noise-aware connectome classifier](methods/fmri.md).
- How the comparison was kept fair: [evaluation protocol](protocol.md).
- The numbers: [results](results.md), and the
  [interactive dashboard](https://mahdisabetkish.github.io/brainssl/).
- Running it yourself: [pipeline, Docker and cluster](reproduce.md), and
  [inference outside Python](deploy.md).

## Repository layout

```
src/brainssl/
  data/          dataset download (parallel, resumable, streamed tar extraction)
  smri/          preprocessing, bilateral U-Net, masking, training, inference, metrics
  fmri/          ABIDE loading, SPD geometry, windows, classifiers, cross-validation, study
  export/        ONNX export, model cards, deployment bundles
  report/        tables and paired tests from run artefacts
  stats.py       intervals and paired tests
rust/            brainssl-infer, a Rust binary for both models (ONNX Runtime)
dashboard/       the static site behind the dashboard link
main.nf, modules/, conf/   the Nextflow pipeline
assets/          run matrices and the pre-registered model choices
docs/            this documentation
```
