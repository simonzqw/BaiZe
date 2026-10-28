#!/usr/bin/env bash
# Evaluate the frozen demonstration checkpoint with English-only console output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA="$ROOT/adamson_smoke.h5ad"
MODEL="$ROOT/checkpoints/adamson_inference_ema.pth"
OUT="$ROOT/results/reproduced_metrics.json"
LOG="$ROOT/results/evaluation.log"

# Download the pinned release asset if needed, then verify its checksum.
bash "$ROOT/download_checkpoint.sh"

for file in "$DATA" "$MODEL" "$ROOT/evaluate_diffusion.py" "$ROOT/models/scerso_diffusion.py" "$ROOT/utils/data_processor.py" "$ROOT/english_console.py"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Missing or empty required file: $file" >&2
        exit 1
    fi
done
mkdir -p "$ROOT/results"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

"$PYTHON" - \
    --data_path "$DATA" \
    --model_path "$MODEL" \
    --batch_size 32 \
    --split_strategy custom \
    --split_col split \
    --task_mode single_gene \
    --perturb_parse_mode raw \
    --background_key cell_context \
    --sample_steps 20 \
    --residual_sample_scale 0.3 \
    --control_match_mode random \
    --control_match_scope global \
    --output_json "$OUT" <<'PY' 2>&1 | tee "$LOG"
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

sys.argv = ["evaluate_diffusion.py", *sys.argv[1:]]
with english_output():
    runpy.run_path("evaluate_diffusion.py", run_name="__main__")
PY

"$PYTHON" check_english.py "$LOG"

"$PYTHON" - "$OUT" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
required = [
    "test_unseen_single_pearson", "test_unseen_single_mse",
    "test_unseen_single_pearson_delta", "test_unseen_perturb_pearson",
    "perturb_pearson_delta",
]
for name in required:
    if name not in metrics:
        raise RuntimeError(f"Missing metric: {name}")
    if not math.isfinite(float(metrics[name])):
        raise RuntimeError(f"Non-finite metric: {name}")
print("PASS: checkpoint loading, inference and metric export")
print("PASS: evaluation log contains English-only console output")
print("Metrics saved to:", path)
PY
