# Data preparation and experiment design: HSPC temporal RNA-state prediction

This note summarizes the recorded Figure 6a–e workflow for time-conditioned RNA prediction from human HSPC multiome data. It describes the path from donor/day Multiome observations to single-cell prediction records and documents what is not recoverable from the archived preprocessing evidence. It intentionally contains no machine-specific paths.

## 1. Original data and prediction task

- **Data source:** human CD34+ haematopoietic stem/progenitor cell (HSPC) time-course multiome, [GEO GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370).
- **Original usable format:** processed donor/day `.h5mu` files with RNA (`mdata.mod['rna']`) and ATAC (`mdata.mod['atac']`) modalities.
- **Donors used in the recorded study:** `13176`, `27678`, `31800`, `32606`.
- **Common modeled days:** **Day 2** as the starting-state RNA, and **Day 3 / Day 7 / Day 10** as target-time conditions. Some source files include Day 4, which is **not** one of these main target days.
- **Cell states:** HSC, EryP, MkP, NeuP, MoP and MasP; rare BP observations were typically excluded from the main visual summaries.

The method predicts Day3/Day7/Day10 RNA separately from a Day2 starting state. It is **not lineage tracking of the same biological cell**, nor iterative Day3→Day7→Day10 autoregression.

## 2. From donor/day Multiome files to model input

1. **Read donor/day multiome data.** Extract the RNA and ATAC modalities from each `.h5mu`, retaining donor, day, cell-state/sample metadata and raw cell identifiers.
2. **Select the modeled time points and consistent RNA features.** Form Day2 starting-state pools and Day3/Day7/Day10 target-state observations. The recorded formal single-cell prediction dataset has **17,865 shared RNA genes**. The exact upstream gene-filtering, normalization, variable-gene selection and scaling implementation has **not** been recovered: do not substitute an arbitrary RNA preprocessing recipe and call it the original method.
3. **Construct state-transition records.** Associate a Day2 RNA input with a target-day RNA observation and the appropriate donor/cell-state/time context. The stored records represent a *conditional* time shift; `TIME_SHIFT` is a computational task label, not an actual CRISPR perturbation. The precise Day2-control sampling/matching algorithm and whether the same Day2 cell is reused have not been confirmed from source code.
4. **Create ATAC conditioning.** The recorded predictive model uses **50-dimensional global ATAC-LSI** context features. Prepare three distinct variants: **RNA-only** (without ATAC conditioning), **shuffled/mismatched ATAC** (negative control), and **correct target-time-matched ATAC**. The shuffle scope and the original LSI fitting/normalization procedure must still be checked against the training-data generator.
5. **Freeze donor-held-out splits and save AnnData.** Preserve the processed RNA records, condition metadata, gene order, ATAC condition variant and `obs['split']`. The three recorded model-input filenames are:

```text
GSE305370 donor/day multiome .h5mu (RNA + ATAC)
  -> donor/day RNA and ATAC extraction, gene alignment and state-transition record building
  -> fixed single-cell leave-one-donor-out (LODO) split
     ├─ gse305370_singlecell_lodo_heldout_13176_rna_only_max500.h5ad
     ├─ gse305370_singlecell_lodo_heldout_13176_shuffled_atac_max500.h5ad
     └─ gse305370_singlecell_lodo_heldout_13176_correct_atac_max500.h5ad
```

The `max500` suffix belongs to the recorded artifact names. Without the original generator, it should **not** be interpreted as proof of an exact per-donor/day/cell-state cap or sampling algorithm.

## 3. Recorded leave-one-donor-out (LODO) split

| Partition | Processed prediction records |
|---|---:|
| Train (source donors 27678, 31800, 32606) | 27,268 |
| Validation (selection from non-test data; exact donor assignment not recovered) | 8,454 |
| **Held-out donor 13176 test** | **8,352** |
| **Total** | **44,074** |

All test records belong to held-out donor **13176**. The three target days contribute **2,628 Day3**, **2,724 Day7** and **3,000 Day10** test records. The **44,074** count describes processed model records, not necessarily distinct original cell barcodes. The source scripts for the exact validation allocation and starting-state control matching have not yet been recovered.

A crucial protocol distinction: **correct-time ATAC** can supply the *target day's ATAC context* during RNA prediction, but it must **not** use the held-out donor's *target-day RNA* as a model input. This is **target-time ATAC-assisted prediction across donors**, not blind forecasting of every modality and not extrapolation to an unseen day label.

## 4. Experiments performed

- Train/evaluate separate **RNA-only**, **shuffled ATAC** and **correct-time ATAC** conditions on the same held-out-donor task; compare Day3, Day7 and Day10 prediction metrics.
- Analyze predicted/true RNA distributions with a selected **2,000-gene** shared visualization matrix (the recorded visualization contains 8,352 rows in each of five plotted groups: Day2 control, true target and three prediction settings).
- Visualize **real** ATAC time structure using a *separate* joint-peak/LSI pipeline; inspect temporal gene programs and cell-state-specific functional enrichment. This is descriptive ATAC analysis, **not prediction of future ATAC peaks**.

**ATAC visualization versus model features:** the recorded ATAC figure aligns 214,241 shared peaks across Day2/3/7/10 from donor 13176, selects informative peaks, runs TF-IDF and a shared 30-component SVD, drops the first component and plots a **29D shared-LSI UMAP**. This **29D visualization representation must not replace the model's 50D global ATAC-LSI conditioning**.

## 5. Validation and outstanding provenance

Verify `obs['donor'] × obs['split']`, source/target time and cell-type distributions, the 17,865-gene order, the presence/absence of `obsm['atac_feat']` for each model setting, and train/test barcode disjointness where required. For shuffled ATAC, record the precise shuffle unit and seed. Retain explicit source-to-control pairing IDs if available, to detect source-state reuse and potential leakage.

The historic training launcher is known as `run_singlecell_lodo_13176_full.sh`, but its complete arguments, validation allocation, gene preprocessing details, control-matching strategy and ATAC shuffle construction have **not** been fully retrieved. The notebook figures and saved evaluation files are evidence of executed analyses; they do not by themselves reconstruct the missing data-generation implementation.
