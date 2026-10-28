#!/usr/bin/env bash
# Retrain a small demonstration checkpoint with English-only console output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA="$ROOT/data/adamson_smoke.h5ad"
OUT="${1:-$ROOT/outputs/smoke_retrain}"

for file in "$DATA" "$ROOT/train_diffusion.py" "$ROOT/models/scerso_diffusion.py" "$ROOT/utils/data_processor.py" "$ROOT/english_console.py"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Missing or empty required file: $file" >&2
        exit 1
    fi
done
if [[ -e "$OUT" ]]; then
    echo "ERROR: Output directory exists; refusing to overwrite: $OUT" >&2
    exit 1
fi
mkdir -p "$OUT"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

"$PYTHON" - \
    --data_path "$DATA" \
    --save_dir "$OUT" \
    --preset none \
    --split_strategy custom \
    --split_col split \
    --task_mode single_gene \
    --perturb_parse_mode raw \
    --background_key cell_context \
    --target_mode delta \
    --epochs 3 \
    --batch_size 64 \
    --timesteps 200 \
    --sample_steps 20 \
    --val_sample_batches 1 \
    --residual_sample_scale 0.3 \
    --control_match_mode random \
    --control_match_scope global \
    2>&1 <<'PY' | tee "$OUT/train.log"
import random
import runpy
import sys

import numpy as np
import torch
from english_console import english_output

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# PyTorch 2.1 compatibility: this is a launcher-only shim.
if not hasattr(torch.amp, "GradScaler"):
    torch.amp.GradScaler = (
        lambda device="cuda", enabled=True:
        torch.cuda.amp.GradScaler(enabled=enabled)
    )

sys.argv = ["train_diffusion.py", *sys.argv[1:]]
with english_output():
    runpy.run_path("train_diffusion.py", run_name="__main__")
PY

"$PYTHON" check_english.py "$OUT/train.log"
echo "PASS: training log contains English-only console output"
echo "Training completed: $OUT"
