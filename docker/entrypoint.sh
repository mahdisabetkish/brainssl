#!/usr/bin/env bash
# Stages as subcommands. Anything not listed here runs as given, so
# `docker run ... python -m brainssl.smri.train finetune --set budget=10pct` works too.
set -euo pipefail

case "${1:-info}" in
  info)
    python - <<'PY'
import platform, torch, brainssl
print(f"brainssl {brainssl.__version__} | python {platform.python_version()} | torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
PY
    ;;
  data)
    shift
    brainssl download abide "$@"
    brainssl download msd "$@"
    brainssl prepare abide
    brainssl prepare msd
    ;;
  smri|fmri|export|prepare|download|report|dashboard)
    exec brainssl "$@"
    ;;
  reproduce)
    # the whole study on the mounted data/ and runs/, in the order of docs/reproduce.md
    python scripts/run_smri.py
    brainssl fmri study --scheme kfold
    brainssl fmri study --scheme loso
    python scripts/run_fmri.py
    brainssl report
    brainssl export all
    exec brainssl dashboard
    ;;
  test)
    shift
    exec python -m pytest -q "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
