# Data preparation and experiment design: Sci-Plex drug perturbation

This note traces the Sci-Plex study data through condition standardization, structure/dose encoding, split construction and Figure 3 evaluation. It distinguishes documented experimental outputs from preprocessing details that remain to be recovered. Machine-specific paths are intentionally omitted.

## 1. Source data and scope

- **Study:** Sci-Plex single-cell chemical perturbation data ([GEO GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944)).
- **Starting experimental AnnData:** \`Sci_Plex.h5ad\`, a previously processed Sci-Plex export, not a claim that raw sequencing FASTQs were directly converted by the repository.
- **Recorded model-ready matrix:** **354,640 cells × 200 genes** from MCF7 (177,730), K562 (89,199) and A549 (87,711); **187 treated-drug labels** plus \`control\`. The selection history of the **200 genes** from earlier data is not fully documented, and these 200 features should not be described as the complete transcriptome.

## 2. Transformation to model-ready AnnData

The repository's preparation entry point is \`scripts/prepare_sciplex_drug.py\`.

1. **Control and drug labels:** use \`condition\` as \`obs['perturbation']\`; consolidate control, vehicle/DMSO and relevant control-indicator columns into integer \`obs['is_control']\`; set control perturbation labels to \`control\`.
2. **Chemical structure:** map \`SMILES\` to \`obs['smiles']\`. The data processor subsequently converts valid structures to **2,048-bit RDKit Morgan fingerprints** (radius 2) for structure-conditioned prediction.
3. **Dose:** parse \`obs['dose']\` as numeric and set vehicle/control dose to zero. The model data processor uses \`log1p(dose)\` followed by dataset-range normalization. Recorded numerical levels are \`0, 10, 100, 1000, 10000\`; **physical dose units must be verified in source metadata** before relabeling plots.
4. **Biological context:** map \`cell_type\` to \`cell_context\` (and \`cell_line\`), with a documented fallback for missing context labels. Construct or retain \`condition_key\` for drug × context × dose and \`drug_key\` for reporting.
5. **Preserve original split semantics:** standardize existing \`split_random\`, \`split_ood\` and other source labels to \`train\`, \`val\`, \`test\`. The test/holdout design comes from the source annotations; a validation subset is sampled from remaining non-control training cells, and controls are kept in training by default. Do **not** replace these partitions with a generic 80/10/10 split.
6. **Save processed input:** keep the RNA expression matrix, standardized condition fields, fixed gene order and split columns in \`Sci_Plex_drug_standardized.h5ad\`. A separate previously prepared \`Sci_Plex_drug_crisp_like_splits.h5ad\` contains compound cell-context × drug holdouts.

\`\`\`text
Sci-Plex study data
  -> Sci_Plex.h5ad (preprocessed study export; earlier 200-gene selection not recovered)
  -> scripts/prepare_sciplex_drug.py
  -> Sci_Plex_drug_standardized.h5ad
       ├─ split_random_std
       └─ split_ood_std
  -> Sci_Plex_drug_crisp_like_splits.h5ad (separately generated; original builder not available)
       ├─ split_crisp_k562_drugood_std
       ├─ split_crisp_a549_drugood_std
       └─ split_crisp_mcf7_drugood_std
\`\`\`

The CRISP-like split file is an **independent recorded artifact**, not a direct output of the current \`prepare_sciplex_drug.py\` script. Its exact generation algorithm and original random seed have not been recovered.

## 3. Recorded evaluation partitions

| Split / setting | Train cells | Validation cells | Test cells | Interpretation |
|---|---:|---:|---:|---|
| \`split_random_std\` | 228,230 | 23,914 | 102,496 | IID/random, with conditions shared across partitions |
| \`split_ood_std\` | 284,554 | 30,172 | 39,914 | Mixed test partition containing both previously seen and unseen drugs |
| CRISP-like K562 | see fixed split file | see fixed split file | 2,573 | Held-out K562 perturbed context × held-out drugs |
| CRISP-like A549 | see fixed split file | see fixed split file | 2,954 | Held-out A549 perturbed context × held-out drugs |
| CRISP-like MCF7 | see fixed split file | see fixed split file | 6,323 | Held-out MCF7 perturbed context × held-out drugs |

The \`split_ood_std\` test partition contains **9 genuinely unseen drugs** (**11,850 cells; 108 drug–dose–context groups**) alongside **178 seen drugs** (**28,064 cells**). Reporting overall OOD test performance as a pure unseen-drug score would therefore be incorrect; the unseen subset must be explicitly filtered.

For CRISP-like compound holdouts, the nine drugs are Alvespimycin, Belinostat, Dacinostat, Flavopiridol, Givinostat, Hesperadin, Quisinostat, TAK-901 and Tanespimycin. They span four numeric nonzero doses (36 drug × dose groups per held-out context). The held-out context contributes **no treated cells to training**, but its untreated controls may be used to define the initial cellular background. Additional source rows may be marked \`ignore\` and must not be silently included.

## 4. Experiments performed

- Compare IID structure-conditioned drug-response prediction with held-out-drug generalization.
- Evaluate compound holdouts of **cell context and drug identity** for K562, A549 and MCF7.
- Analyze drug/dose groups, targeted pathways and response magnitude; the molecular target/pathway annotations serve mainly for **post hoc interpretation**, not as chemical-structure model inputs.
- Compute predictions using an actual trained checkpoint and compare predicted versus observed RNA changes; use \`scripts/evaluate_drug_response.py\` and maintain cell-level and group-level metric definitions.

## 5. Important checks and limitations

Inspect \`adata.obs\` to verify control counts, drug identities, SMILES validity, dose values, context labels and split counts. Ensure that no purported unseen drug has treated training examples, that each testing context has accessible training controls, and that each CRISP-like holdout has the expected drug–dose combinations. Historical evaluations were run with saved model predictions, but matching the originally used model/checkpoint and reconstructing the earliest 200-gene selection and CRISP-like split-generation code still require verification. Do not label drug-dose values with unverified physical units.
