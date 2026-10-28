# Installation and evaluation of the BaiZe Adamson reviewer smoke test

This directory is an English-language **software smoke-test demonstration**. It is not the full Figure 2b benchmark or an example of strong unseen-perturbation performance.

## Public artifacts

- **Source, data and scripts:** [BaiZe/reviewer_demo/Adamson](https://github.com/simonzqw/BaiZe/tree/main/reviewer_demo/Adamson)
- **Example checkpoint:** [adamson-smoke-v1 pre-release](https://github.com/simonzqw/BaiZe/releases/tag/adamson-smoke-v1)
- **Direct checkpoint download:** [adamson_inference_ema.pth](https://github.com/simonzqw/BaiZe/releases/download/adamson-smoke-v1/adamson_inference_ema.pth)

The checkpoint is 91,528,898 bytes (approximately 87.29 MiB). Its GitHub-published SHA-256 digest is:

```text
3dad6cde7a2530dd4d2736e352b0f2f291ef7cf5e9617bcf0e3380eaa2d2ef89
```

Do not commit this model artifact to the Git history: the launcher downloads it into `checkpoints/adamson_inference_ema.pth` when needed.

## Clone and evaluate

On a Linux machine with Python 3.10, an appropriate PyTorch installation, and the other dependencies described in `ENVIRONMENT.md`:

```bash
git clone https://github.com/simonzqw/BaiZe.git
cd BaiZe/reviewer_demo/Adamson
bash verify_package.sh
bash run_evaluation.sh
```

Both verification and evaluation automatically call `download_checkpoint.sh`. The first run needs Internet access plus `curl` or `wget`; the downloaded binary must match its pinned SHA-256 digest. A mismatched existing file is not overwritten automatically.

To download the checkpoint explicitly:

```bash
bash download_checkpoint.sh
```

For an already activated compatible environment, the scripts use `python`. You may specify a different interpreter:

```bash
PYTHON=/path/to/python bash verify_package.sh
PYTHON=/path/to/python bash run_evaluation.sh
```

Expected evaluation artifacts are `results/reproduced_metrics.json` and an English-only `results/evaluation.log`. The recorded local example metrics are in `results/expected_metrics.json`.

## Optional training demonstration

```bash
bash run_training_smoke.sh
```

This runs three training epochs with a predefined perturbation-disjoint split. The output is written to a new `outputs/smoke_retrain/` directory. It will not overwrite a prior run. The training does not recreate the exact published example checkpoint.

## Dataset licensing and scientific limitations

The 2,020-cell subset is publicly available as `adamson_smoke.h5ad` in this folder. Its original accession, attribution and redistribution rights still require separate verification. Hosting it publicly does not establish those permissions.

The original model source is kept byte-for-byte identical to the tested archive. The English console wrapper translates legacy log messages without changing computational operations.

A successful software smoke test does not reproduce the manuscript's reported full Figure 2b results. The example checkpoint's delta-response correlations are close to zero. For source hashes and release status, see `SOURCE_PROVENANCE.md` and `PUBLICATION_STATUS.md`.
