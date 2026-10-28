# Adamson reviewer demo: publication status

**Status as of 2026-10-28: Documentation and launchers uploaded; public runnable package incomplete.**

This folder is part of the BaiZe manuscript's `reviewer_demo/` workspace. The authors have successfully run the software smoke test locally. The public repository does not yet contain every artifact needed to reproduce that test from a new clone.

## Available on GitHub

- English README, environment notes, installation guidance, and requirements.
- Training, evaluation, and verification launcher scripts.
- English-language console/log adapter and validation script.
- Recorded demonstration metrics from the lightweight EMA checkpoint.

## Still required for independent execution

- `data/adamson_smoke.h5ad` (approximately 21.31 MiB): upload only after confirming the source attribution and redistribution permissions.
- `checkpoints/adamson_inference_ema.pth` (approximately 87.29 MiB): store as a versioned release asset or other stable public download, not as a normal repository commit.
- **The exact model/data-processing source used in the successful server test**, including `train_diffusion.py`, `evaluate_diffusion.py`, `models/`, and `utils/`. The tested archived source may differ from the current repository-root implementation; do not silently substitute it.
- A clean-clone test using the public artifact URLs and published source.

## Scientific scope

The successful local test used 2,020 K562 cells and 5,060 genes, with 1,700/160/160 train/validation/test cells; 86 distinct perturbations; 3 training epochs; 200 diffusion timesteps; and a compact EMA checkpoint. It established that the software pathway executes, **not** that the manuscript's full Figure 2b metrics are reproduced.

The observed delta-response correlations were close to zero. Do not present them as evidence of competitive zero-shot perturbation prediction.

## Artifact provenance

The public `results/expected_metrics.json` contains metrics supplied by the local evaluation log. Its `model_path` field was changed from the original absolute server path to the portable relative path `checkpoints/adamson_inference_ema.pth`. Numerical values are unchanged.

## Publication checklist

1. Verify parent dataset attribution, license, and subset redistribution rights.
2. Publish the exact tested code files rather than another repository revision.
3. Publish or link the dataset and inference checkpoint.
4. Update README quick-start commands to fetch any externally hosted artifacts.
5. Run `verify_package.sh` and `run_evaluation.sh` from a fresh checkout.
6. Only then describe the public folder as an independently executable smoke-test package.

The pretrained-model sources and availability restrictions are documented separately at [../PRETRAINED_MODELS.md](../PRETRAINED_MODELS.md).
