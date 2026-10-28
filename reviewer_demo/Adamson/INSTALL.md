# Installation and package update

This update supplies **English-only reviewer-facing documents, scripts, and console messages**. It keeps the verified AnnData input, pretrained example checkpoint, original research implementation, and numeric evaluation settings unchanged.

The archive contains **no** `.h5ad` or `.pth` files. It should be extracted into an existing `reviewer_package/` directory with the dataset, checkpoint, core model code, and `results/expected_metrics.json` already in place.

## Install

Copy the ZIP to the parent directory of the existing `reviewer_package/` directory and extract its files into `reviewer_package/`.

```bash
BASE=/home/zqw/data1/Baize_reviewer_demo/Adamson
python3 -m zipfile -e "$BASE/BaiZe_Adamson_English_v2.zip" "$BASE/reviewer_package"
chmod +x "$BASE/reviewer_package/"*.sh
cd "$BASE/reviewer_package"
```

## Verify and evaluate

```bash
PYTHON=/home/zqw/conda/envs/cellwhisperer/bin/python bash verify_package.sh
PYTHON=/home/zqw/conda/envs/cellwhisperer/bin/python bash run_evaluation.sh
```

The evaluation prints and records English messages, writes `results/reproduced_metrics.json`, and checks `results/evaluation.log` for unexpected non-English characters.

## Retrain (optional)

```bash
PYTHON=/home/zqw/conda/envs/cellwhisperer/bin/python bash run_training_smoke.sh
```

The training output defaults to `outputs/smoke_retrain/` and is never allowed to overwrite an existing directory. Its log is checked for English-only output.

## Scientific integrity and limitations

- The text-translation layer only affects displayed console messages, not numerical calculations.
- The underlying research source files are retained unchanged and may still include legacy non-English comments and print literals. The launchers translate those messages for the reviewer-facing console and logs.
- If an unknown non-English source message occurs, the launcher fails visibly rather than dropping a warning or traceback.
- A successful smoke test demonstrates that the software pipeline can run, **not** that the manuscript's Figure 2b performance has been reproduced.
- The parent dataset provenance and redistribution permissions must be checked before releasing the data subset publicly.