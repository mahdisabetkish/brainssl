# Reproduce

## With Nextflow

`main.nf` runs both studies from the raw download to the results tables, the
exported models and the dashboard data.

```bash
# everything, on one GPU machine
nextflow run mahdisabetkish/brainssl -profile docker,gpu

# data already on disk, one study only
nextflow run . -profile docker,gpu --raw data/raw --tracks smri

# on a Slurm cluster with Apptainer
nextflow run . -profile slurm,apptainer,gpu --cluster_options '--account=abc123'

# pick up where an interrupted run stopped
nextflow run . -profile docker,gpu -resume
```

The run matrices are data: `assets/smri_runs.csv` lists every pretraining and
fine-tuning run with its label budget, initialisation and number of steps, and
`assets/fmri_methods.csv` lists the deep fMRI baselines. A run that starts from
a pretrained checkpoint waits only for that pretraining run. The linear fMRI
models all come from one `FMRI_STUDY` process per protocol, which computes the
features of each split once and fits every model and every C on them.

`-profile test` runs every stage on a miniature synthetic dataset with tiny
networks in a few minutes on a CPU; CI does this on every push, after a stub run
that checks the channel logic alone. Nextflow 24.10 or later and Java 17 or later
are needed.

## Without Nextflow

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-lock.txt
pip install -e .

brainssl download abide && brainssl download msd     # about 9.5 GB
brainssl prepare abide && brainssl prepare msd

python scripts/run_smri.py          # segmentation matrix, resumable
brainssl fmri study --scheme kfold  # every linear connectome model
brainssl fmri study --scheme loso
python scripts/run_fmri.py          # deep connectome baselines
brainssl report                     # tables and tests
brainssl export all                 # ONNX, refitted classifier, model cards
brainssl dashboard                  # data behind the dashboard
```

The download stage reads the 7.6 GB MSD archive as parallel byte ranges and
unpacks it while it streams, so the archive itself never touches the disk; an
interrupted download resumes at the first member it had not finished.

## Docker

Two training images are built from one Dockerfile, CPU and CUDA 12.6, and a
small third image carries only the Rust binary and the exported models.

```bash
docker run --rm --gpus all --shm-size=4g \
  -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" \
  mahdisabetkish/brainssl:gpu smri finetune --set budget='"10pct"'

docker run --rm -v "$PWD/scans:/scans" mahdisabetkish/brainssl-infer \
  segment --model /models/segmentation --input /scans/case.nii.gz --output /scans/case_seg.nii.gz
```

`--shm-size` matters: data-loading workers hand batches over through shared
memory and Docker's default of 64 MB is too small for 3D patches.

## Compute

The reported numbers were produced on one workstation.

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 Ti, 11 GB |
| CPU | Intel Xeon E5-2697 v4, 18 cores |
| RAM | 32 GB |
| Software | Windows 10, Python 3.11.8, PyTorch 2.7.1 with CUDA 12.6 |

A segmentation step (two 160 x 128 x 128 patches) takes about one second on this
GPU, so the pretraining run takes under three hours and the whole segmentation
matrix about a day. The fMRI study runs on the CPU. Pascal GPUs have no tensor
cores; mixed precision is used anyway because it halves activation memory and
speeds up the memory-bound parts of the network.

## Determinism

Every entry point takes a seed for Python, NumPy and PyTorch. cuDNN autotuning
is left on for speed, so repeated runs agree in distribution rather than bit for
bit. The data splits are deterministic: the segmentation split is stored in
`data/processed/msd/splits.json`, and the connectome splits are a function of
the seeds alone.
