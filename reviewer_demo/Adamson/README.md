# BaiZe: Adamson Perturb-seq Reviewer Demonstration

> **Publication status (2026-10-28):** The exact tested source code, 2,020-cell demonstration dataset, recorded metrics and English scripts are public. The 3-epoch EMA example checkpoint is distributed separately as a [GitHub Release asset](https://github.com/simonzqw/BaiZe/releases/tag/adamson-smoke-v1) (pre-release). The launchers now download and SHA-256-verify that checkpoint automatically. The original local test passed; a separate fresh-clone GPU execution remains to be verified. See [`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md).

## Purpose and scope

This package is a **software reproducibility smoke test** for the BaiZe conditional diffusion framework, using a small subset of real Adamson Perturb-seq data. It provides executable training and evaluation entry points, a demonstration checkpoint, and recorded test metrics.

**This is not a reproduction of the full Adamson benchmark or the quantitative results reported in Figure 2b of the manuscript.** The demonstration checkpoint was trained for only three epochs and does not use external pretrained gene embeddings. The observed delta-response correlations are close to zero and must not be interpreted as evidence of strong unseen-perturbation performance.

## Data and splits

The prepared `adamson_smoke.h5ad` file contains **2,020 cells and 5,060 genes** from K562 cells, representing **86 single-gene perturbation categories** plus a control category. Using fixed random seeds, 20 cells were sampled per perturbation and 300 control cells were sampled.

| Partition | Number of cells | Perturbation categories |
| --- | ---: | --- |
| Training | 1,700 | 70 gene perturbations plus 300 controls |
| Validation | 160 | 8 held-out gene perturbations |
| Test | 160 | 8 held-out gene perturbations |

Perturbation categories are disjoint across the three partitions. All selected control cells are assigned to training, and training controls also serve as the reference control pool during held-out evaluation. The original source file used to generate this subset was `perturb_processed_scerso_compat_ctxfix.h5ad` (68,603 cells, 5,060 genes). The subset has been uploaded to the public repository. Parent-dataset attribution, a public source accession/download URL, and redistribution permission still require verification; public availability alone does not establish redistribution rights.

## Package contents

```text
Adamson/
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── run_evaluation.sh
├── run_training_smoke.sh
├── verify_package.sh
├── english_console.py
├── check_english.py
├── download_checkpoint.sh
├── INSTALL.md
├── .gitignore
├── adamson_smoke.h5ad
├── checkpoints/
│   └── adamson_inference_ema.pth
├── results/
│   └── expected_metrics.json
├── models/
├── utils/
├── train_diffusion.py
└── evaluate_diffusion.py
```

For a fresh public checkout, the dataset, verified source code, English launchers, and recorded metrics are included. The model checkpoint is hosted as a [versioned GitHub Release attachment](https://github.com/simonzqw/BaiZe/releases/download/adamson-smoke-v1/adamson_inference_ema.pth), not in Git history. `download_checkpoint.sh`, `verify_package.sh`, and `run_evaluation.sh` automatically retrieve the checkpoint when missing and verify its pinned SHA-256 digest; an existing file with the wrong digest is rejected.

## Environment

The workflow was exercised on Ubuntu Linux, Python 3.10, NVIDIA GeForce RTX 3090, PyTorch 2.1.1+cu121, Scanpy 1.9.5, and AnnData 0.10.2. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for details and the compatibility notes. Use a compatible CUDA-enabled PyTorch installation for GPU execution.

RDKit is not needed for RNA-only Adamson testing when `utils/data_processor.py` uses the optional RDKit import fallback that was applied to the local smoke-test copy. RDKit is required for drug-structure prediction.

## Quick-start evaluation

After cloning the [BaiZe repository](https://github.com/simonzqw/BaiZe) and activating a compatible Python environment:

```bash
cd BaiZe/reviewer_demo/Adamson
bash verify_package.sh
bash run_evaluation.sh
```

These commands automatically download the example checkpoint from GitHub Releases if it is absent; a working Internet connection and `curl` or `wget` are required for this first download. You can also download it explicitly using `bash download_checkpoint.sh`. The SHA-256 expected for the release asset is `3dad6cde7a2530dd4d2736e352b0f2f291ef7cf5e9617bcf0e3380eaa2d2ef89`.

**Versioning note:** Clone the current `main` branch for the documented launchers. The GitHub Release tag was created before the automatic-download script was added, so the tag's auto-generated source archive is not the current recommended entry point. The Release **asset** is the pinned checkpoint; the documented source and scripts are in `main`.

To specify the Python interpreter explicitly:

```bash
PYTHON=/path/to/environment/bin/python bash run_evaluation.sh
```

The launcher uses the [public EMA checkpoint](https://github.com/simonzqw/BaiZe/releases/tag/adamson-smoke-v1), stored locally at `checkpoints/adamson_inference_ema.pth`, and evaluates the 160 test cells with the predefined `obs['split']` assignments, 20 diffusion sampling steps, `residual_sample_scale=0.3`, random control matching, and an EMA-adjusted model state. **Do not add `--use_ema` when loading this compact checkpoint**, because the EMA parameters have already been merged into `model_state_dict`.

The evaluation output is written to `results/reproduced_metrics.json`, and its English-only terminal log is saved in `results/evaluation.log`. A runtime localization check translates inherited source-code messages without changing the model. The launcher checks that expected metric fields are present and finite. It does not require bitwise equality with the recorded metrics because stochastic sampling and runtime implementation details can affect numerical results.

## Retraining smoke test

```bash
bash run_training_smoke.sh
```

Or specify a fresh output directory:

```bash
bash run_training_smoke.sh /path/to/new/output_dir
```

This trains on the provided subset for three epochs using 200 diffusion timesteps, batch size 64, delta-space prediction, and `residual_sample_scale=0.3`. The script does not overwrite an existing output directory. On PyTorch 2.1, the launcher provides a compatibility wrapper for `torch.amp.GradScaler`; the training source code remains unchanged.

The translated training log is saved in `outputs/smoke_retrain/train.log` and automatically checked for English-only content. Unrecognized non-English output triggers an explicit diagnostic rather than being silently removed.

The retrained checkpoint is expected to **differ** from the provided example checkpoint, especially in stochastic numerical outputs. This small training exercise validates the software path, not paper-level accuracy.

## Recorded evaluation metrics

The following are observed values from the **included lightweight EMA checkpoint** with `residual_sample_scale=0.3` (the values were obtained in the user's smoke-test run, not a full-paper experiment):

| Metric | Observed value |
| --- | ---: |
| Single-cell Pearson | 0.6896156 |
| Single-cell MSE | 0.2423829 |
| Single-cell delta Pearson | 0.0036817 |
| Single-cell Top20 delta Pearson | -0.1074548 |
| Perturbation-level Pearson | 0.8038622 |
| Perturbation-level delta Pearson | 0.0086967 |
| Perturbation-level Top20 delta Pearson | -0.2335715 |

The complete recorded JSON is `results/expected_metrics.json`. **The original training checkpoint and the compact EMA checkpoint did not yield identical evaluations** in the comparison run; the recorded values above belong specifically to the compact checkpoint. No numerical equivalence claim is made.

## Verified functionality and limitations

The smoke-test run demonstrated data loading, metadata and control-pool processing, training, checkpoint saving, independently loading a checkpoint, inference over the held-out test partition, and JSON metric export.

This does **not** establish the reported Figure 2b performance. Full benchmark reproduction requires the original full data, validated preprocessing, split protocol, model configuration, training duration, pretrained-feature setup where applicable, and matching metric definitions. The parent dataset's public provenance and redistribution rights still require verification for the already uploaded `adamson_smoke.h5ad`.

## English-only reviewer-facing output

All distributed documentation, setup instructions, launcher messages, and the generated console logs are in English. This update uses `english_console.py` to translate legacy logging text at runtime, leaving the verified training and evaluation logic unchanged. The inherited research-source files may contain original comments and string literals; those source files are not rewritten by this update. Unexpected non-English runtime messages cause an explicit English diagnostic and must be resolved before publication.

## Main code repository

https://github.com/simonzqw/BaiZe