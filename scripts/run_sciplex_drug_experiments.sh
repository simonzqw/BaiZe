#!/usr/bin/env bash
set -euo pipefail
python scripts/prepare_sciplex_drug.py --input /root/autodl-fs/project3/data/Sci_Plex.h5ad --output /root/autodl-fs/project3/data/Sci_Plex_drug_standardized.h5ad --force_controls_train
# random
python train_diffusion.py --data_path /root/autodl-fs/project3/data/Sci_Plex_drug_standardized.h5ad --save_dir checkpoints_sciplex_random_structure --task_mode drug --split_strategy custom --split_col split_random_std --background_key cell_context --context_col cell_context --perturb_col perturbation --smiles_col smiles --dose_col dose --control_col is_control --drug_condition_mode structure --target_mode delta --batch_size 4096 --preset vnext --amp
python scripts/evaluate_drug_response.py --data_path /root/autodl-fs/project3/data/Sci_Plex_drug_standardized.h5ad --model_path checkpoints_sciplex_random_structure/best_model.pth --config_path checkpoints_sciplex_random_structure/config.json --split_col split_random_std --output_json checkpoints_sciplex_random_structure/drug_eval.json --output_csv checkpoints_sciplex_random_structure/drug_group_metrics.csv
