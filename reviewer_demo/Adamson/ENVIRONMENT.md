# BaiZe Adamson Smoke Test: Tested Environment

## Observed execution environment

| Component | Observed version / specification |
| --- | --- |
| Operating system | Ubuntu Linux |
| Python | 3.10 (Conda `cellwhisperer`) |
| GPU | NVIDIA GeForce RTX 3090, 23.56 GiB reported |
| PyTorch | 2.1.1+cu121 |
| Scanpy | 1.9.5 |
| AnnData | 0.10.2 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| tqdm | 4.70.0 |

These are **observed installed versions**, not a guarantee that every installation combination will resolve cleanly on another machine. Reuse an isolated environment; do not upgrade an existing research environment solely to run the demo.

## Using an existing environment

Run `bash verify_package.sh` to confirm the files and basic module imports. For example, in the previously tested setup:

```bash
PYTHON=/home/zqw/conda/envs/cellwhisperer/bin/python bash verify_package.sh
PYTHON=/home/zqw/conda/envs/cellwhisperer/bin/python bash run_evaluation.sh
```

The exact `/home/zqw/...` path is from the originating server and should be replaced on other machines.

## Installing dependencies on a different machine

Create a fresh Python 3.10 environment. Install a GPU-enabled or CPU-compatible PyTorch 2.1.1 distribution suitable for the machine (the tested GPU build was PyTorch 2.1.1+cu121), then install the other pinned dependencies from `requirements.txt`. The requirements file records direct dependencies observed for this demo; additional transitive dependencies may be resolved by the package installer.

`torch.amp.GradScaler('cuda', ...)` is not available in the tested PyTorch 2.1.1 build. The included `run_training_smoke.sh` injects a compatibility wrapper around `torch.cuda.amp.GradScaler`. This affects only the training launcher and does not edit BaiZe source files.

The RNA-only smoke test uses an optional `rdkit` import in `utils/data_processor.py`, applied to the local packaged copy. **Do not assume the unmodified upstream source works without RDKit.** Drug experiments that compute SMILES fingerprints require RDKit.

An unrelated Scanpy `pkg_resources` deprecation warning was observed but did not prevent successful training and evaluation.

## Commands and model download

```bash
bash verify_package.sh
bash run_evaluation.sh
bash run_training_smoke.sh
```

The first two commands automatically download the [example EMA checkpoint](https://github.com/simonzqw/BaiZe/releases/tag/adamson-smoke-v1) if it is missing. The `download_checkpoint.sh` helper verifies the SHA-256 digest published with the GitHub Release and refuses to overwrite a mismatched local checkpoint. On first use, Linux must have `curl` or `wget`, `sha256sum`, and Internet access.

`run_evaluation.sh` saves `results/reproduced_metrics.json`. `run_training_smoke.sh` writes a fresh run to `outputs/smoke_retrain` by default. Output values are stochastic and may differ slightly between machines.

## English-language output

The launchers provide English reviewer-facing logs via `english_console.py`, and `check_english.py` verifies generated log files. This runtime localization has no effect on model tensors, checkpoints, or metric calculations.