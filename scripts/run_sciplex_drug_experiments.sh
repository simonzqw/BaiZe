#!/usr/bin/env bash
# Sci-Plex drug-response training and evaluation. Run from the BaiZe repository root.
set -euo pipefail

DATA_DIR="${SCIPLEX_DATA_DIR:-/root/autodl-fs/project3/data}"
RAW_H5AD="${SCIPLEX_RAW_H5AD:-$DATA_DIR/Sci_Plex.h5ad}"
DATA_H5AD="${SCIPLEX_DATA_H5AD:-$DATA_DIR/Sci_Plex_drug_standardized.h5ad}"
SPLIT_COL="${SCIPLEX_SPLIT_COL:-split_random_std}"
OUT_DIR="${SCIPLEX_OUT_DIR:-checkpoints_sciplex_random_structure}"

# Set SCIPLEX_PREPARE=0 when using an existing preprocessed file (e.g. CRISP-like splits).
if [[ "${SCIPLEX_PREPARE:-1}" == "1" ]]; then
  python scripts/prepare_sciplex_drug.py \
    --input "$RAW_H5AD" --output "$DATA_H5AD" --force_controls_train
fi
if [[ ! -s "$DATA_H5AD" ]]; then
  echo "Sci-Plex input not found or empty: $DATA_H5AD" >&2
  exit 1
fi

# A separate output directory must be used for each fixed-split experiment.
python train_diffusion.py \
  --data_path "$DATA_H5AD" --save_dir "$OUT_DIR" \
  --task_mode drug --drug_condition_mode structure --target_mode delta \
  --split_strategy custom --split_col "$SPLIT_COL" \
  --background_key cell_context --context_col cell_context \
  --perturb_col perturbation --smiles_col smiles --dose_col dose \
  --control_col is_control --preset vnext --epochs 50 \
  --batch_size 4096 --timesteps 1000 --sample_steps 50 \
  --timestep_sampler loss-second-moment --guidance_scale 1.0 \
  --cond_dropout 0.1 --early_stop_metric composite \
  --score_w_delta 1.0 --score_w_top20p 0.2 \
  --score_w_top20mse 0.02 --amp

CHECKPOINT="$OUT_DIR/best_model_diff.pth"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Expected trained checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
python scripts/evaluate_drug_response.py \
  --data_path "$DATA_H5AD" --model_path "$CHECKPOINT" \
  --split_col "$SPLIT_COL" --batch_size 2048 \
  --sample_steps 50 --guidance_scale 1.0 \
  --residual_sample_scale 0.0 --weights ema --seed 42 \
  --output_json "$OUT_DIR/drug_eval_model.json" \
  --output_csv "$OUT_DIR/drug_group_metrics_model.csv" \
  --pred_npz "$OUT_DIR/predictions_test.npz"
