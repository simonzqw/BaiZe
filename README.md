# Response-space state-transition modeling predicts cellular responses across perturbation contexts

**BaiZe: a multi-view conditional diffusion framework for cellular-response prediction**

<p align="center">
  <img src="docs/assets/baize_framework_latest.png" width="100%" alt="BaiZe framework">
</p>

BaiZe models a cellular response as a **conditional state transition**. Given an unperturbed single-cell RNA profile and a perturbation context, the framework predicts the perturbation-induced expression shift and reconstructs the post-intervention state:

\[
\hat{x}_{\mathrm{pert}} = x_{\mathrm{ctrl}} + \widehat{\Delta x}.
\]

The control transcriptome is encoded as a cellular-background representation, while genetic interventions, drug structure and dose, temporal or state information, species context, and optional chromatin accessibility are mapped to a unified condition representation. A conditional diffusion decoder then iteratively generates the expression-change vector \(\widehat{\Delta x}\). This design provides a shared modeling interface for heterogeneous perturbation settings while preserving the biological context of the starting cell state.

## Framework scope

| Modeling setting | Inputs | Predicted output | Main analysis supported |
|---|---|---|---|
| Genetic perturbation | Control RNA and one or more perturbed genes | Single-cell expression change and perturbed RNA state | Single-gene, combinatorial and higher-order perturbation prediction |
| Chemical perturbation | Control RNA, molecular structure, dose and cellular context | Drug-induced expression change | Unseen-drug, held-out-cell-context and dose-response analysis |
| ATAC-guided prediction | Control RNA, perturbation condition and optional ATAC context | Chromatin-informed RNA response | RNA-only versus RNA+ATAC comparison, promoter masking and peak-level attribution |
| Temporal state transition | Earlier control state, target time or state and optional matched ATAC | Future RNA state | Time-course prediction, lineage progression and donor-held-out evaluation |
| Cross-species transfer | Source-species perturbation data, target-species control state and ortholog-aligned features | Target-species perturbation response | Human-to-mouse zero-shot transfer and few-shot target-species adaptation |
| Downstream interpretation | Predicted expression matrices and derived response signatures | Genes, pathways, regulatory regions and projected phenotypes | Functional enrichment, morphology projection and evidence-linked interpretation |

For SNP-informed applications, BaiZe uses a variant-associated candidate gene to define a perturbation task and predicts the downstream transcriptional consequences of altered gene activity. This analysis does not constitute direct allele-specific SNP-effect prediction.

## Model overview

BaiZe contains four main components:

1. **Cellular-background encoder**: maps the control RNA profile to a latent representation of the pre-intervention cell state.
2. **Condition encoder**: represents gene perturbations, multi-gene combinations, drug structure, dose, time, state, species context and optional ATAC features in a shared condition space.
3. **Conditional diffusion decoder**: predicts the perturbation-induced expression shift through iterative denoising with classifier-free guidance.
4. **Output reconstruction**: adds the predicted expression shift to the control state to obtain the simulated post-intervention transcriptome.

The default target is the expression-change space (`--target_mode delta`), which separates perturbation effects from the high baseline similarity shared by control and perturbed transcriptomes.

## Repository structure

| Path | Description |
|---|---|
| `train_diffusion.py` | Unified training entry point for the conditional diffusion model |
| `evaluate_diffusion.py` | Single-cell and perturbation-level evaluation |
| `predict_diffusion.py` | Perturbation inference, multi-gene composition and interpolation |
| `visualize_diffusion.py` | Response visualization and combinatorial perturbation analysis |
| `models/scerso_diffusion.py` | BaiZe model definition |
| `models/diffusion_core.py` | Diffusion process, sampling and denoising utilities |
| `utils/data_processor.py` | AnnData loading, split construction, control pools and condition fields |
| `scripts/` | Drug, ATAC, temporal, cross-species and perturbation-analysis workflows (including `scripts/evaluate_drug_response.py`) |
| `scripts_morphology/` | Export and conversion utilities for transcriptome-guided morphology projection |
| `multiperturb_seq/` | MultiPerturb-seq RNA+ATAC processing and analysis |
| `norman_combo_gears_style/` | Combinatorial perturbation splits and residual modeling utilities |

## Installation

BaiZe was developed with Python 3.10 and PyTorch. The core workflow requires PyTorch, Scanpy/AnnData, NumPy, SciPy, pandas, scikit-learn, matplotlib and tqdm. RDKit is required when drug structures are encoded from SMILES strings.

```bash
conda create -n baize python=3.10 -y
conda activate baize
pip install torch scanpy anndata numpy scipy pandas scikit-learn matplotlib seaborn tqdm rdkit
```

## Input organization

The main training input is an AnnData (`.h5ad`) object. Gene order must remain consistent across control and perturbed matrices. The required observation fields depend on the experiment, but the framework uses a common organization:

| Field | Purpose |
|---|---|
| `perturbation` | Gene, multi-gene or drug perturbation label |
| `is_control` | Control-state indicator |
| `split` | Optional predefined train, validation and test assignment |
| `cell_context` | Cell line, cell type, state, donor or other biological background |
| `smiles` | Drug molecular structure for structure-conditioned prediction |
| `dose` | Drug concentration or continuous treatment intensity |
| `atac_feat` | Optional aligned chromatin-accessibility representation |

Control labels such as `control`, `ctrl`, `DMSO` and `vehicle` should be harmonized before model training. Cross-dataset analyses additionally require a consistent gene order and, where applicable, one-to-one ortholog alignment.

## Experiment-specific training and evaluation

The examples below are intended to be run **from the repository root**. Processed AnnData files and trained checkpoints are not bundled with the repository; adjust the paths to your data environment. The commands assume that the preprocessing and fixed splits for each experiment have already been prepared. For drug experiments, the example paths refer to the original AutoDL layout.

### 1. Genetic perturbation: Adamson UPR triple-gene holdout

Train on control, single- and double-gene interventions, and evaluate the fully held-out `ATF6+EIF2AK3+ERN1` combination (14,675 cells, 5,002 genes; split stored in `obs['split']`).

```bash
DATA=/root/autodl-fs/triple_perturb/adamson_upr/processed/adamson_upr_triple_clean_split_hvg5000_log1p_dense.h5ad
OUT=adamson_upr_triple/checkpoints_hvg5000_dense_final

python train_diffusion.py \
  --data_path "$DATA" --save_dir "$OUT" \
  --split_strategy custom --split_col split \
  --perturb_parse_mode raw --background_key cell_line \
  --target_mode delta --epochs 50 --batch_size 256 \
  --timesteps 1000 --sample_steps 50 --val_sample_batches 3 \
  --patience 12 --control_match_scope global \
  --control_prototype_mode topk_weighted --control_match_k 32 \
  --cond_dropout 0.1 --lambda_topde 0.2 \
  --lambda_delta_corr 0.1 --lambda_centroid 0.1 --topde_k 50

python evaluate_diffusion.py \
  --data_path "$DATA" --model_path "$OUT/best_model_diff.pth" \
  --output_json "$OUT/eval_triple_best_diff.json" \
  --split_strategy custom --split_col split \
  --perturb_parse_mode raw --background_key cell_line \
  --target_mode delta --sample_steps 50 --guidance_scale 1.0 \
  --control_match_scope global \
  --control_prototype_mode topk_weighted --control_match_k 32
```

The source experiment also records residual-sampling sensitivity analyses; an evaluation run must use the residual-sampling setting of the corresponding checkpoint/report to reproduce a specific number. For a small, separately validated end-to-end example, see the [Adamson reviewer demonstration](reviewer_demo/Adamson/README.md).

### 2. Chemical perturbation: Sci-Plex (Figure 3)

Drug prediction uses a **structure-conditioned** diffusion model with SMILES/Morgan fingerprints, numerical dose, control-cell RNA and cell context. The experiment includes IID/random, held-out-drug OOD, and held-out-cell-context × unseen-drug comparisons. The following values are the recorded main training configuration:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `task_mode` | `drug` | `drug_condition_mode` | `structure` |
| `target_mode` | `delta` | `epochs` | `50` |
| `timesteps` | `1000` | `sample_steps` | `50` |
| `timestep_sampler` | `loss-second-moment` | `guidance_scale` | `1.0` |
| `cond_dropout` | `0.1` | `amp` | enabled |
| `early_stop_metric` | `composite` | `split_strategy` | `custom` |
| `score_w_delta` | `1.0` | `score_w_top20p` | `0.2` |
| `score_w_top20mse` | `0.02` | `background_key` | `cell_context` |
| `smiles_col` | `smiles` | `dose_col` | `dose` |
| `control_col` | `is_control` | `context_col` | `cell_context` |

**Train:**

```bash
DATA=/root/autodl-fs/project3/data

# Set the processed dataset and split for each experiment.
# 1) IID/random:         Sci_Plex_drug_standardized.h5ad     split_random_std
# 2) Drug OOD:           Sci_Plex_drug_standardized.h5ad     split_ood_std
# 3) K562 compound OOD:  Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_k562_drugood_std
# 4) A549 compound OOD:  Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_a549_drugood_std
# 5) MCF7 compound OOD:  Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_mcf7_drugood_std

train_drug() {
  local label="$1" data="$2" split="$3"
  python train_diffusion.py \
    --data_path "$DATA/$data" \
    --save_dir "checkpoints_sciplex_${label}_structure" \
    --task_mode drug --drug_condition_mode structure --target_mode delta \
    --split_strategy custom --split_col "$split" \
    --background_key cell_context --context_col cell_context \
    --perturb_col perturbation --smiles_col smiles \
    --dose_col dose --control_col is_control \
    --preset vnext --epochs 50 --batch_size 4096 \
    --timesteps 1000 --sample_steps 50 \
    --timestep_sampler loss-second-moment \
    --guidance_scale 1.0 --cond_dropout 0.1 \
    --early_stop_metric composite \
    --score_w_delta 1.0 --score_w_top20p 0.2 \
    --score_w_top20mse 0.02 --amp
}

train_drug random Sci_Plex_drug_standardized.h5ad split_random_std
train_drug ood Sci_Plex_drug_standardized.h5ad split_ood_std
train_drug crisp_k562 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_k562_drugood_std
train_drug crisp_a549 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_a549_drugood_std
train_drug crisp_mcf7 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_mcf7_drugood_std
```

Each call trains a **separate** model in its own checkpoint directory, with `best_model_diff.pth` as the selected checkpoint. For an existing trained model, skip its training command and use its saved checkpoint rather than overwriting it.

**Evaluate:** `scripts/evaluate_drug_response.py` loads the saved checkpoint (raw weights or EMA), performs model inference on the requested test split, and reports cell-level and equal-weighted drug–dose–context group-level metrics.

```bash
# Same DATA variable as above.
evaluate_drug() {
  local label="$1" data="$2" split="$3"
  local out="checkpoints_sciplex_${label}_structure"
  python scripts/evaluate_drug_response.py \
    --data_path "$DATA/$data" \
    --model_path "$out/best_model_diff.pth" \
    --split_col "$split" --batch_size 2048 \
    --sample_steps 50 --guidance_scale 1.0 \
    --residual_sample_scale 0.0 --weights ema --seed 42 \
    --output_json "$out/drug_eval_model.json" \
    --output_csv "$out/drug_group_metrics_model.csv" \
    --pred_npz "$out/predictions_test.npz"
}

evaluate_drug random Sci_Plex_drug_standardized.h5ad split_random_std
evaluate_drug ood Sci_Plex_drug_standardized.h5ad split_ood_std
evaluate_drug crisp_k562 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_k562_drugood_std
evaluate_drug crisp_a549 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_a549_drugood_std
evaluate_drug crisp_mcf7 Sci_Plex_drug_crisp_like_splits.h5ad split_crisp_mcf7_drugood_std

# Strict unseen-drug subset of the OOD test split:
python scripts/evaluate_drug_response.py \
  --data_path "$DATA/Sci_Plex_drug_standardized.h5ad" \
  --model_path checkpoints_sciplex_ood_structure/best_model_diff.pth \
  --split_col split_ood_std --batch_size 2048 --sample_steps 50 \
  --guidance_scale 1.0 --residual_sample_scale 0.0 \
  --weights ema --seed 42 --unseen_only \
  --output_json checkpoints_sciplex_ood_structure/drug_eval_unseen_only.json \
  --output_csv checkpoints_sciplex_ood_structure/drug_group_metrics_unseen_only.csv
```

The evaluation makes **real checkpoint predictions**. Its output includes Top20/Top50 delta Pearson and MSE, opposite-direction rates and `predictions_test.npz` when requested. The `--residual_sample_scale` option explicitly fixes the sampling contribution: `0.0` evaluates the model's predicted mean delta, while nonzero settings additionally use the sampled diffusion residual. This choice must match the protocol used for a reported result. The OOD `split_ood_std` test partition contains both seen and unseen compounds; use `--unseen_only` to measure the latter specifically. Drug concentration units must be confirmed from the dataset metadata before interpreting dose-response plots.

### 3. Human-to-mouse cross-species transfer (Figure 5)

The cross-species experiment aligns one-to-one human/mouse orthologs, learns a source-perturbation residual-transfer model, and evaluates ARID1A/PDCD1 on mouse TIL control backgrounds. This module has **its own** training, inference and evaluation entry points:

```bash
V2=/root/autodl-fs/k_spices/processed/cross_species_v2
COMBINED=/root/autodl-fs/k_spices/processed/cross_species_human_train_mouse_test_atac_replicate256.h5ad
MOUSE=/root/autodl-fs/k_spices/processed/mouse_til_perturb_cross_available_strict_atac_replicate256.h5ad
VOCAB=/path/to/verified_perturb_vocab.txt

python scripts/cross_species_build_pseudobulk.py \
  --data_path "$COMBINED" \
  --out_npz "$V2/cross_species_pseudobulk_bootstrap128_atac_controlatac_top20w10.npz" \
  --split_col split --train_split_value train \
  --species_col species --human_value human --atac_key atac_feat \
  --atac_source control --bootstrap_reps 128 --top_de_n 20 --top_de_weight 10

python scripts/cross_species_train_residual.py \
  --pseudobulk_npz "$V2/cross_species_pseudobulk_bootstrap128_atac_controlatac_top20w10.npz" \
  --save_dir checkpoints_cross_species_residual_REBUILT \
  --perturb_vocab_path "$VOCAB" --use_atac --learn_perturb_alpha \
  --rank 64 --hidden_dim 512 --residual_scale 0.01 \
  --epochs 100 --batch_size 64 --lr 3e-5 --seed 42

python scripts/cross_species_infer_residual.py \
  --combined_h5ad "$COMBINED" --mouse_h5ad "$MOUSE" \
  --ckpt checkpoints_cross_species_residual_REBUILT/best_model_cross_species_residual.pth \
  --out_dir preds_cross_species_REBUILT \
  --perturbations ARID1A PDCD1 --atac_key atac_feat \
  --contextwise --aggregate_contexts

python scripts/evaluate_cross_species_mouse.py \
  --mouse_h5ad "$MOUSE" --pred_npz preds_cross_species_REBUILT/predictions.npz \
  --out_json preds_cross_species_REBUILT/eval_all_gene.json
python scripts/evaluate_cross_species_context_preds.py \
  --mouse_h5ad "$MOUSE" --pred_npz preds_cross_species_REBUILT/predictions.npz \
  --out_dir preds_cross_species_REBUILT/context_eval
```

The training example is a **source-code-compatible reconstruction** rather than a verified copy of the original checkpoint configuration. The 5%/10% few-shot preparation and adaptation procedure requires its corresponding original scripts and saved splits; global all-gene evaluation is not the manuscript's Top20 metric.

### 4. Time-conditioned state prediction (Figure 6)

The GSE305370 HSPC experiment predicts Day 3/7/10 RNA from Day 2 RNA, comparing RNA-only, shuffled ATAC and time-matched ATAC on held-out donor 13176. The three processed `.h5ad` settings are:

```text
time_analisy/h5ad_for_diffusion_singlecell_lodo/
  gse305370_singlecell_lodo_heldout_13176_rna_only_max500.h5ad
  gse305370_singlecell_lodo_heldout_13176_shuffled_atac_max500.h5ad
  gse305370_singlecell_lodo_heldout_13176_correct_atac_max500.h5ad
```

The historical training entry point is `run_singlecell_lodo_13176_full.sh` (which calls `train_diffusion.py`); run it from the original experimental archive after checking its data, donor split and settings. Its exact arguments are not fully preserved in the public repository. A compatible *interface* for evaluating an existing correct-ATAC checkpoint is:

```bash
TIME_ROOT=/root/autodl-tmp/scERso/time_analisy
python evaluate_diffusion.py \
  --data_path "$TIME_ROOT/h5ad_for_diffusion_singlecell_lodo/gse305370_singlecell_lodo_heldout_13176_correct_atac_max500.h5ad" \
  --model_path "$TIME_ROOT/model_outputs_singlecell_lodo/heldout_13176_correct_atac_full_residual03/best_model_diff.pth" \
  --split_strategy custom --split_col split --atac_key atac_feat \
  --output_json "$TIME_ROOT/model_outputs_singlecell_lodo/correct_atac_eval.json"
```

The original evaluation settings should be recovered from the saved training configuration and prediction logs before comparing this command's output with the reported Figure 6 metrics.

## Evaluation

BaiZe is evaluated primarily in the expression-change space. Reported measures include whole-transcriptome Pearson correlation, expression-change Pearson correlation, Top-k expression-change Pearson correlation, Top-k mean squared error, recovery of strongly responsive genes and the fraction of top response genes predicted in the opposite direction. Metrics can be computed at both the single-cell level and after aggregation by perturbation, dose, cell context, time point or species-specific state.

## Data sources

| Dataset or resource | BaiZe analysis | Accession or source |
|---|---|---|
| Sci-Plex | Chemical structure- and dose-conditioned drug-response prediction | [GEO: GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) |
| MultiPerturb-seq RNA+ATAC | ATAC-guided perturbation prediction, promoter peak masking and peak-level attribution | [GEO: GSE277747](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277747) |
| Human CD8 T-cell perturbation data | Source-species training for human-to-mouse response transfer | [GEO: GSE119450](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119450) |
| Additional human CD8 T-cell perturbation resource | Additional reference dataset | [GEO: GSE218988](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218988) |
| Mouse tumour-infiltrating T-cell RNA / ATAC | Target-species zero-shot and few-shot evaluation | [RNA: GSE203592](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203592); [ATAC: GSE203591](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203591) |
| Related mouse perturbation resource | Additional reference dataset | [GEO: GSE203593](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203593) |
| HSPC time-course multiome | Temporal RNA-state prediction with optional matched ATAC context | [GEO: GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370) |
| Replogle K562 CRISPRi Perturb-seq | SNP-informed HIF1A case study and related perturbation-response analyses | [Processed Perturb-seq release on Figshare](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387); [Genome-Wide Perturb-Seq portal](https://gwps.wi.mit.edu/) |
| Ensembl BioMart | Human-mouse one-to-one ortholog mapping and gene annotation | [Ensembl BioMart](https://www.ensembl.org/biomart/martview) |
| UCSC Genome Browser | Reference genome coordinates and genomic-region annotation | [UCSC Genome Browser](https://genome.ucsc.edu/) |

## Cross-species workflow

The recommended cross-species workflow aligns human and mouse genes in a shared one-to-one ortholog space, estimates source-species perturbation effects, calibrates them with the target-species control background and evaluates the resulting response across target-species cell states.

```text
Human perturbation data
        ↓
One-to-one ortholog alignment
        ↓
Target-species control-state calibration
        ↓
Mouse perturbation-response prediction
        ↓
Zero-shot evaluation or few-shot target-species adaptation
```

The main scripts are `scripts/cross_species_diagnose_data.py`, `scripts/cross_species_build_pseudobulk.py`, `scripts/cross_species_train_residual.py`, `scripts/cross_species_infer_residual.py`, `scripts/evaluate_cross_species_mouse.py` and `scripts/evaluate_cross_species_context_preds.py`.

## BaiZe-Agent

<p align="center">
  <a href="https://enjoyed-sitemap-tables-underwear.trycloudflare.com/"><strong>Open the BaiZe-Agent interactive demo</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/simonzqw/BaiZe"><strong>View the BaiZe repository</strong></a>
</p>

<p align="center">
  <img src="docs/assets/baize.png" width="100%" alt="BaiZe-Agent Explorer interface">
</p>

BaiZe-Agent is a natural-language interface for exploring results produced by the BaiZe single-cell perturbation modelling framework. It does not retrain BaiZe or generate new cellular states during a query. Instead, it retrieves structured evidence from indexed CSV, JSON and result files, matches related figures through a figure registry, and returns evidence-grounded explanations with traceable source records.

### Current capabilities

| Capability | Supported exploration |
|---|---|
| Cross-species perturbation responses | Human-to-mouse prediction metrics, response genes, state-specific patterns and related figures |
| Combinatorial perturbations | Single-, dual- and triple-gene response analysis, including higher-order combinations |
| Drug and dose responses | Unseen-drug performance, dose-dependent responses and cell-context comparisons |
| Temporal prediction | RNA-only and RNA+ATAC predictions across time points, states and held-out donors |
| ATAC-informed interpretation | ATAC-corrected genes, promoter masking results and peak-level attribution |
| Functional interpretation | Gene-response, pathway-enrichment, differential-expression and metric queries |
| Evidence and figure retrieval | Indexed result-file lookup and dynamic display of registered figures |
| Report generation | Saving answers, evidence records and related figures as PDF reports |

### Example questions

- How did BaiZe perform on unseen drugs?
- Show the PDCD1 cross-species volcano plot.
- Which genes were corrected by ATAC for BAZ1B?
- How did correct-time ATAC improve prediction at Day7?
- Which genes responded most strongly to ATF6+EIF2AK3+ERN1?

### Evidence handling

Numerical values are read from indexed result files whenever available, and related images are retrieved through the figure registry. The interface distinguishes retrieved evidence from narrative interpretation and retains the source paths used to construct each response. Prediction agreement, differential expression, enrichment and attribution analyses provide model-supported evidence, but do not by themselves establish causality, pathway activation or direct molecular regulation.

The interface provides a new-exploration view, recent session history, saved results and an integrated help page. The public demonstration is available at [BaiZe-Agent Explorer](https://enjoyed-sitemap-tables-underwear.trycloudflare.com/).

## Reproducibility

Model checkpoints, large processed matrices and generated result files are not generally stored in the root repository. The commands above identify the expected data files, model weights, split names and evaluator interfaces. To reproduce a reported manuscript value, use the corresponding fixed cell/gene split, matching checkpoint and original evaluation configuration. The [Adamson reviewer demonstration](reviewer_demo/Adamson/README.md) is a runnable software smoke test, not the complete paper benchmark.
