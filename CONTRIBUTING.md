# Contributing

Issues and pull requests are welcome. The project is small, so a short note on
what you want to change before a large pull request saves everyone time.

## Setup

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e ".[dev]"
pre-commit install            # optional: ruff and file hygiene on each commit
```

## Checks

```bash
ruff check src tests scripts bin
pytest -q
cargo test --manifest-path rust/Cargo.toml     # only if you touch rust/
```

CI runs the same checks, an end-to-end pass on synthetic data
(`bin/make_test_data.py`), the Nextflow test profile and the Rust parity
checks. All of it runs on a CPU in a few minutes, so you do not need the
datasets or a GPU to work on the code.

## Conventions

- Python 3.10 or later, `ruff` for style, 120 columns, type hints on public
  functions.
- A change that affects a reported number needs a new run through the
  pipeline and a regenerated `docs/results.md`, never a hand edit.
- Choices that decide a comparison (regularisation, window count, splits) are
  fixed before the reported runs and recorded in `assets/`. Changing one after
  the fact needs a note in the pull request saying so.
- Prose in the documentation is plain and specific; claims stay in proportion to the
  evidence, and limitations are stated where the result is.
