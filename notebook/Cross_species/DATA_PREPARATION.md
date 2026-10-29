# Data preparation and experiment design: human-to-mouse perturbation transfer

This note documents the recorded source-to-target data construction for the BaiZe human CD8 T-cell to mouse tumour-infiltrating T-cell (TIL) experiment (Figure 5). It describes the residual-transfer experimental line, rather than the archived earlier cross-species diffusion experiments. No local storage paths are given.

## 1. Biological source datasets

| Input | Public source | Use |
|---|---|---|
| Human CD8 T-cell perturbation RNA | [GEO GSE119450](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119450) | Source control and **ARID1A / PDCD1** perturbation responses, including NoStim and Stim contexts |
| Mouse MC38 TIL perturbation RNA | [GEO GSE203592](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203592) | Mouse untreated controls for prediction and mouse perturbed RNA for held-out evaluation/adaptation |
| Mouse accessibility data | [GEO GSE203591](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203591) | Unperturbed ATAC-derived context features |
| Human–mouse one-to-one ortholog mapping | Gene-symbol mapping table (exact historic version to be preserved) | Shared expression feature order |

The recorded mouse TIL contexts are C1-S1pr1, C2-ISG, C3-41BB, C4-ProgEx, C5-TermEx and C6-Cycling.

## 2. Data construction, step by step

1. **Prepare human and mouse AnnData objects.** Preserve expression, perturbation labels, context labels and sample/cell identifiers. Normalize control aliases (e.g. CTRL, CTRL1, vehicle) to the common `control` label.
2. **Align genes by one-to-one orthologs.** Using a mapping table with `human_gene_symbol` and `mouse_gene_symbol`, match source and target features, reorder both matrices identically and label the shared columns with human gene symbols. The recorded aligned input uses **15,520 genes**. A matching column count is not enough: identical **gene order** is required.
3. **Build a joint dataset with fixed species split.** `scripts/prepare_cross_species_human_mouse.py` selects human cells for training/validation and labels mouse cells as the held-out target species. `obs` contains `species`, `perturbation`, `cell_context` and `split`.
4. **Create target-species ATAC context features.** `scripts/build_cross_species_atac_map.py` constructs promoter-associated accessibility features from annotation/ATAC inputs; `scripts/attach_atac_replicate_features.py` attaches control ATAC replicate-level features as `obsm['atac_feat']` (recorded 256-dimensional variant). These are **replicate-level/context-derived features**, not proof of per-cell paired ATAC. The exact historical replicate assignments and all source-processing settings need archival verification.
5. **Make human source pseudobulk training rows.** `scripts/cross_species_build_pseudobulk.py` groups **training human** data by cell context and perturbation. For each group, bootstrap control and perturbed cells (recorded **128 repeats**, up to **256** cells from each pool with replacement), then compute `control_mean`, `target_mean`, `true_delta`, `source_delta`, `residual_target`, control ATAC-derived `atac_mean` and gene weights. The recorded Top20 response-gene weighting is **10**.
6. **Train and predict in the shared gene space.** `scripts/cross_species_train_residual.py` fits the independent `CrossSpeciesResidualPredictor` on source pseudobulk rows. `scripts/cross_species_infer_residual.py` combines learned source response, mouse control RNA, ATAC context (when enabled) and a perturbation label to produce mouse-state predictions. The target-species perturbed RNA is reserved for evaluation in zero-shot.

```text
Human RNA + mouse RNA + one-to-one ortholog table
  -> matched 15,520-gene AnnData with species / context / split
  -> human perturbation-control bootstrap pseudobulk .npz
  -> cross-species residual checkpoint
  -> predictions on mouse control backgrounds
  -> comparison to held-out mouse perturbed RNA

Mouse ATAC + gene annotations -> control-context ATAC feature bank -> optional .obsm['atac_feat']
```

Important: the exact normalization of the earliest human/mouse expression inputs, ortholog mapping file revision, and original source feature generation must be verified from the frozen processed artifacts. The public alignment script alone does not establish all upstream processing choices.

## 3. Source–target partition design

**Zero-shot:** mouse perturbed responses are not used for training or response-supervised tuning. Mouse control RNA, pre-existing context labels and *unperturbed* ATAC-derived features may provide the target background.

| Original cell-level split | Cell count |
|---|---:|
| Human train | 22,585 |
| Human validation | 2,509 |
| Mouse test total | 29,702 |
| **Combined** | **54,796** |

The mouse set contains **18,569 controls**, **4,668 ARID1A-treated** and **6,465 PDCD1-treated** cells. The alignment script records an approximately 90%/10% human train/validation split using `seed=42`, with mouse labeled `test`. The residual trainer additionally makes an **85%/15% internal split of bootstrap pseudobulk rows**. These two split levels must not be confused.

**Few-shot adaptation:** a fixed-seed selection of mouse perturbed cells is used for target-species adaptation, and the remainder is held out for response evaluation.

| Adaptation setting | Mouse perturbed adaptation | Mouse perturbed held-out test | Mouse control pool |
|---|---:|---:|---:|
| 5%, seed 42 | 557 | 10,576 | 18,569 |
| 10%, seed 42 | 1,114 | 10,019 | 18,569 |

Historical notes describe stratification by perturbation × mouse context, but the **original few-shot cell-ID manifest and split-generation scripts are not available in the current public tree**. The 5% and 10% samples must not be assumed nested. Likewise, the full historic few-shot checkpoint-initialization procedure has not been reconstructed from the public training script. Do not present a newly trained model as an exact reproduction of the historic adaptation run without verification.

## 4. Experiments performed

- Compare human→mouse **zero-shot** response prediction for ARID1A and PDCD1, including RNA-only and ATAC-conditioned variants.
- Evaluate **5% and 10% few-shot** mouse adaptation on separate remaining perturbed-cell tests.
- Analyze global and C1–C6 context-level predicted-versus-true expression changes, Top-k response genes and direction consistency, with source-delta and additional baselines as applicable.
- Use `scripts/evaluate_cross_species_mouse.py` and `scripts/evaluate_cross_species_context_preds.py` for basic global/context-level checks. Any Top20 manuscript result must be calculated under the documented gene-selection and evaluation protocol, not inferred from these scripts' all-gene outputs.

## 5. Quality-control requirements

Verify one-to-one mapping and identical `var_names`; counts for both species, each perturbation and all contexts; nonempty mouse control pools; 256-dimensional ATAC features when enabled; and absence of perturbed mouse response supervision in zero-shot. For 5%/10% experiments, compare **cell IDs** (not just cell counts) to prove disjoint adaptation and held-out sets. Freeze the original ortholog table, expression normalization provenance, ATAC replicate mapping and split manifests before claiming exact end-to-end reproducibility.
