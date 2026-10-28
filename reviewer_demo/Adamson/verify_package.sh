#!/usr/bin/env bash
# Check layout, imports, data dimensions and English-only distributable files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# A fresh checkout downloads the pinned model automatically.
bash "$ROOT/download_checkpoint.sh"

for file in \
    "$ROOT/README.md" \
    "$ROOT/ENVIRONMENT.md" \
    "$ROOT/INSTALL.md" \
    "$ROOT/requirements.txt" \
    "$ROOT/run_evaluation.sh" \
    "$ROOT/run_training_smoke.sh" \
    "$ROOT/download_checkpoint.sh" \
    "$ROOT/english_console.py" \
    "$ROOT/check_english.py" \
    "$ROOT/adamson_smoke.h5ad" \
    "$ROOT/checkpoints/adamson_inference_ema.pth" \
    "$ROOT/results/expected_metrics.json" \
    "$ROOT/train_diffusion.py" \
    "$ROOT/evaluate_diffusion.py" \
    "$ROOT/models/scerso_diffusion.py" \
    "$ROOT/models/diffusion_core.py" \
    "$ROOT/utils/data_processor.py"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Missing or empty file: $file" >&2
        exit 1
    fi
done
bash -n "$ROOT/run_evaluation.sh"
bash -n "$ROOT/run_training_smoke.sh"
cd "$ROOT"
"$PYTHON" check_english.py
"$PYTHON" - <<'PY'
import anndata as ad
import torch
from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor
from english_console import english_text

assert "Loading data" in english_text("Loading data")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
a = ad.read_h5ad("adamson_smoke.h5ad", backed="r")
try:
    assert a.n_obs == 2020, a.n_obs
    assert a.n_vars == 5060, a.n_vars
    assert set(a.obs["split"].astype(str)) == {"train", "val", "test"}
    print("Dataset:", a.n_obs, "cells x", a.n_vars, "genes")
finally:
    a.file.close()
print("PASS: package structure, imports, and data checks")
PY
