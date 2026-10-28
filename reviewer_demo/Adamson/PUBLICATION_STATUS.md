# Adamson reviewer demo: publication status

**Status as of 2026-10-28: Tested source, demonstration dataset and versioned EMA checkpoint are publicly accessible. A fresh-clone GPU test and parent-dataset license/provenance verification remain outstanding.**

This folder is part of the BaiZe manuscript's `reviewer_demo/` workspace. The authors have successfully run the software smoke test locally. The exact tested training, evaluation, model and data-processing source files have now been uploaded, and the 2,020-cell demonstration dataset is now public at `adamson_smoke.h5ad`. The EMA inference checkpoint is now published at [GitHub Release adamson-smoke-v1](https://github.com/simonzqw/BaiZe/releases/tag/adamson-smoke-v1). The verification and evaluation scripts automatically download and checksum-verify it. A clean-clone GPU execution has not yet been independently confirmed.

## Available on GitHub

- English README, environment notes, installation guidance, and requirements.
- Training, evaluation, and verification launcher scripts.
- English-language console/log adapter and validation script.
- Recorded demonstration metrics from the lightweight EMA checkpoint.
- Uploaded dataset at `adamson_smoke.h5ad` (22,344,537 bytes / approximately 21.31 MiB). Parent-dataset attribution and redistribution permission still require review.
- Versioned EMA inference checkpoint: [GitHub Release asset](https://github.com/simonzqw/BaiZe/releases/download/adamson-smoke-v1/adamson_inference_ema.pth) (91,528,898 bytes; SHA-256 `3dad6cde7a2530dd4d2736e352b0f2f291ef7cf5e9617bcf0e3380eaa2d2ef89`).
- Automatic model download and integrity verification through `download_checkpoint.sh`.
- Exact tested Python source: `train_diffusion.py`, `evaluate_diffusion.py`, `models/scerso_diffusion.py`, `models/diffusion_core.py`, `utils/data_processor.py`, `utils/diffusion_schedule.py`, `utils/emb_loader.py`. See `SOURCE_PROVENANCE.md` for SHA verification.

## Still required for independent execution

- A clean-clone test using the public artifact URLs and published source.

## Scientific scope

The successful local test used 2,020 K562 cells and 5,060 genes, with 1,700/160/160 train/validation/test cells; 86 distinct perturbations; 3 training epochs; 200 diffusion timesteps; and a compact EMA checkpoint. It established that the software pathway executes, **not** that the manuscript's full Figure 2b metrics are reproduced.

The observed delta-response correlations were close to zero. Do not present them as evidence of competitive zero-shot perturbation prediction.

## Artifact provenance

The public `results/expected_metrics.json` contains metrics supplied by the local evaluation log. Its `model_path` field was changed from the original absolute server path to the portable relative path `checkpoints/adamson_inference_ema.pth`. Numerical values are unchanged.

## Publication checklist

1. Verify parent dataset attribution, license, and subset redistribution rights.
2. Confirm that the seven uploaded source blobs match the tested archive (recorded in `SOURCE_PROVENANCE.md`).
3. Confirm that the Release asset and dataset are downloadable from a fresh machine.
4. Run `verify_package.sh` and `run_evaluation.sh` from a fresh checkout on a compatible GPU-equipped environment.
5. Record the clean-clone outcome, software versions and checksum results.
6. Only then describe the public folder as an independently verified standalone smoke-test package.

The pretrained-model sources and availability restrictions are documented separately at [../PRETRAINED_MODELS.md](../PRETRAINED_MODELS.md).
