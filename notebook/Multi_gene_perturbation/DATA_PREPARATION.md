# Data preparation and experiment design: Adamson UPR triple perturbation

This note describes how the Adamson UPR Perturb-seq dataset was converted into the BaiZe multi-gene experiment input. It summarizes recorded preprocessing and fixed splits; the executable training/evaluation entry points are described in the repository root README. No machine-specific paths are needed.

## 1. Source data

- **Study:** Adamson et al., *Cell* (2016), CRISPR interference (CRISPRi) Perturb-seq of the unfolded protein response in K562 cells ([GEO GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546); [scPerturb / Zenodo 7041849](https://zenodo.org/records/7041849)).
- **Downloaded input:** `AdamsonWeissman2016_GSM2406677_10X005.h5ad`; this is the **10X005** sample, not 10X001 or 10X010.
- **Initial AnnData:** 15,006 cells × 32,738 genes, with expression in `adata.X` and perturbation annotations in `adata.obs`.

## 2. From source AnnData to model-ready AnnData

1. **Choose the experimental conditions.** Keep negative controls; single-gene CRISPRi of ATF6, PERK and IRE1; all three pairwise combinations; and the triple combination. Exclude 331 cells with missing, invalid or unrelated perturbation annotations.
2. **Harmonize labels.** Merge both `3x_neg_ctrl_pMJ144` controls into `control`; rename `PERK` to `EIF2AK3`, `IRE1` to `ERN1`; represent each pair and the triple consistently as `ATF6+EIF2AK3`, etc. Retain the original label as `perturbation_raw`.
3. **Restrict genes.** Select approximately 5,000 highly variable genes (HVGs), explicitly retaining `ATF6`, `EIF2AK3` and `ERN1` even if omitted by HVG ranking. The recorded result has **5,002 genes**; the exact original HVG flavor and random state have not been fully recovered. The final gene order should therefore be read from the processed file rather than recomputed and assumed identical.
4. **Normalize expression.** Preserve pre-normalization counts in `adata.layers['counts']`; apply Scanpy `normalize_total(target_sum=1e4)` and `log1p` to the selected expression matrix; store the model input `adata.X` as dense `float32`.
5. **Freeze split metadata.** Store standardized perturbation labels and `obs['split']` alongside the gene order in the processed `.h5ad` file.

Recorded intermediate/final dataset names (file names only):

```text
AdamsonWeissman2016_GSM2406677_10X005.h5ad
  -> adamson_upr_triple_clean_split.h5ad
  -> adamson_upr_triple_clean_split_hvg5000.h5ad
  -> adamson_upr_triple_clean_split_hvg5000_log1p_dense.h5ad
```

The final model input contains **14,675 cells × 5,002 genes**. Key fields include `obs['perturbation']`, `obs['perturbation_raw']`, `obs['cell_context']`, `obs['cell_line']` and `obs['split']`; `var_names` determines the fixed gene order.

## 3. Fixed training, validation and test split

The test task is **higher-order combinatorial generalization**: individual target genes are seen in training, while their *triple combination* is entirely unseen.

| Perturbation condition | Train | Validation | Test |
|---|---:|---:|---:|
| control | 3,028 | 337 | 0 |
| ATF6 | 1,522 | 169 | 0 |
| EIF2AK3 | 1,404 | 156 | 0 |
| ERN1 | 1,516 | 169 | 0 |
| ATF6 + EIF2AK3 | 1,443 | 160 | 0 |
| ATF6 + ERN1 | 1,304 | 145 | 0 |
| EIF2AK3 + ERN1 | 1,466 | 163 | 0 |
| **ATF6 + EIF2AK3 + ERN1** | **0** | **0** | **1,693** |
| **Total** | **11,683** | **1,299** | **1,693** |

The seven lower-order conditions were split approximately 90%/10% into train/validation. The complete triple condition was reserved for test. The exact historic split seed is not documented; reproduce published comparisons with the **frozen `obs['split']`**, not a newly generated random split.

## 4. Experiments performed

- Train BaiZe's conditional diffusion model on control, single- and double-gene response data.
- Predict the held-out `ATF6+EIF2AK3+ERN1` post-perturbation RNA state from control RNA and the triple-gene condition.
- Evaluate whole-expression and perturbation-delta agreement, including Top-k response genes; examine diffusion-residual sampling settings and visualize predicted responses.
- This experiment tests **generalization to an unseen combination**, not unseen genes or a definitive measurement of biological synergy.

## 5. Validation and reproducibility checks

Before use, verify `adata.shape == (14675, 5002)`; confirm `X` is normalized dense `float32`; confirm the three genes exist in `var_names`; reproduce the table above from `pd.crosstab(adata.obs['perturbation'], adata.obs['split'])`; and check that **all** test rows are the triple perturbation. Retain cell IDs, gene order, source-data accession, normalization record and a checksum for the final processed AnnData. The original HVG selection settings and split seed require archival verification for *from-scratch, byte-identical* reconstruction.
