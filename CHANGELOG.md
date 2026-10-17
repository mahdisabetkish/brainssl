# Changelog

## Unreleased

- Two studies on public data: tumour segmentation with a bilateral U-Net on MSD
  Task01, and autism classification from resting-state connectomes on ABIDE I.
- Nextflow pipeline, Docker images, GitHub Actions workflows, a static
  dashboard, and `brainssl-infer`, a Rust binary for both models.
- Interpretability and trust audit for both models (calibration, selective
  prediction, subgroup and shortcut checks, attributions, perturbation tests),
  with written pass/fail rules in `docs/trust.md`.
- Results of the full run matrix, the audit, and the dashboard data.
