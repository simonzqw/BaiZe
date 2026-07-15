# scERso conditional diffusion model (Conditional Diffusion for Single-cell Response)

The current **main line of this warehouse has been switched to the conditional diffusion model**, and the old MLP/Transformer training route (`train.py`) is no longer recommended.

It is currently recommended to use:
- Training: `train_diffusion.py`
- Evaluation: `evaluate_diffusion.py`
- Inference/Combination/Interpolation: `predict_diffusion.py`
- Visualization: `visualize_diffusion.py`

---

## 1. Current mainline capabilities

### 1.1 Conditional diffusion backbone
- Background-effect decoupling: `z_bg` (background) and `z_eff` (effect) are encoded separately.
- Target mode: `target_mode = target | delta`.
- Sampling and guidance: supports classifier-free guidance, DDIM step control, and EMA weight evaluation.

### 1.2 Dual task mode (key point)
Use `--task_mode` to distinguish two types of tasks to avoid "mismatch between task definition and data":

1. `single_gene`
   - Suitable for single gene perturbation tasks such as Adamson
   - Condition fields: `perturb_gene_idx`, `is_control`

2. `translation`
   - Suitable for two-condition translation (such as day4 -> day6)
   - Condition fields: `condition_id`, `source_flag`

### 1.3 Data and control/reference mechanism
- Support `split_strategy = random | perturbation | custom`
- Under the perturbation zero-shot setting, val/test reuses the train control bank to avoid crashes without control split
- Support `control_match_mode`, `control_prototype_mode`, `control_prototype_temp`

---

## 2. Project structure (related to the current main line)

- `train_diffusion.py`: Conditional diffusion training entrance (main entrance)
- `evaluate_diffusion.py`: Evaluation entrance (single-cell + perturbation-level indicators)
- `predict_diffusion.py`: Single perturbation/combination perturbation prediction and latent interpolation trajectory output
- `visualize_diffusion.py`: Combined disturbance analysis plot and diagnostic visualization
- `models/scerso_diffusion.py`: Conditional diffusion model definition
- `models/diffusion_core.py`: Diffusion process implementation
- `utils/data_processor.py`: h5ad reading, partitioning, control pool, condition field construction
- `docs/diffusion_methodology.md`: Methodology Description

> Old route files (such as `train.py`, `evaluate_metrics.py`, `visualize.py`) are retained for historical comparison only and are not used as current recommended paths.

---

## 3. Environment and dependencies

suggestion:
- Python 3.8+
- PyTorch
- scanpy / anndata
- numpy / scipy / pandas
- scikit-learn
- matplotlib / seaborn
- rdkit (only when using the SMILES drug feature)

And recommended settings:

```bash
export OMP_NUM_THREADS=1
```

---

## 4. Training (main entrance)

## 4.1 Adamson（single_gene）

```bash
python train_diffusion.py \
  --data_path /path/to/adamson/perturb_processed.h5ad \
  --save_dir ./checkpoints_adamson_single_gene \
  --task_mode single_gene \
  --split_strategy perturbation \
  --preset vnext \
  --amp
```

If there are combination perturbation tags such as `double_...`, `triple_...` or `GENE1+GENE2+GENE3` in the training data, multi-gene tag analysis can be turned on:

```bash
python train_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --save_dir ./checkpoints_combo_diffusion \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --preset vnext \
  --amp
```

## 4.2 day4/day6（translation）

```bash
python train_diffusion.py \
  --data_path /path/to/day4_to_day6_diffusion.h5ad \
  --save_dir ./checkpoints_day4_day6_translation \
  --task_mode translation \
  --split_strategy custom \
  --split_col split \
  --atac_key atac_feat \
  --preset vnext \
  --amp
```

> Quick smoke available `--preset smoke`.

---

## 5. Evaluation

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --output_json ./checkpoints_xxx/eval_metrics.json
```

A three-gene combination case can be additionally evaluated (if there is a corresponding combination label in h5ad, the true mean indicator of the case will be output at the same time):

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --cell_line K562 \
  --combo_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --output_json ./checkpoints_xxx/eval_metrics_triple.json
```

The translation data can be changed to:

```bash
--task_mode translation --split_strategy custom --split_col split
```

---

## 6. Reasoning and Visualization

### 6.1 Prediction/Combination/Interpolation

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --latent_mode adaptive \
  --save_dir ./pred_out
```

Three-gene perturbation case (just add a third gene to the original two-gene command):

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_dir ./pred_out_triple
```

### 6.2 Visualization

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --save_path ./combo_report.png
```

Three-gene combination visualization:

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_path ./triple_combo_report.png
```

---

## 7. FAQ

### Q1: `adata.obs Missing custom partition column: split`
You used `split_strategy=custom`, but there is no `obs['split']` in the data. Can be changed to:

```bash
--split_strategy perturbation
```

Or prepare the `split` column in h5ad first.

### Q2: `--split_strategy perturbation` is clearly transmitted, but the log shows custom
`--preset` will now only overwrite "not explicitly set" parameters; explicit parameters will be retained. If the error persists, please confirm that parameters are not passed repeatedly on the command line.

### Q3: val/test reports that the control pool is empty
Under perturbation zero-shot, val/test reuses the train control bank. If an error is still reported, it is usually because the training set itself does not have control samples, and you need to check the original data first.

---

## 8. Cross-species perturbation prediction module

This module is independent from the diffusion mainline.

### 8.1 Legacy scripts

- `scripts/prepare_mouse_context.py`
- `scripts/train_cross_species_ctx.py`
- `scripts/cross_species_infer_ctx.py`

### 8.2 Recommended v2 workflow

1. Diagnose data: `scripts/cross_species_diagnose_data.py`
2. Build bootstrap pseudo-bulk: `scripts/cross_species_build_pseudobulk.py`
3. Train residual model: `scripts/cross_species_train_residual.py`
4. Run context-wise inference: `scripts/cross_species_infer_residual.py`
5. Evaluate: `scripts/evaluate_cross_species_mouse.py`, `scripts/evaluate_cross_species_context_preds.py`
