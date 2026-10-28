# Source provenance for the Adamson reviewer smoke test

This folder contains the **exact seven Python source files** from the user-supplied archive `BaiZe_Adamson_source_only.tar.gz`, generated from the successfully tested local `reviewer_package/` directory on 2026-10-28.

The source files were published without altering their model, data-processing, training, evaluation or logging logic. In particular, these are not silently substituted with the possibly divergent files in the BaiZe repository root.

## Git blob SHA-1 verification

| Relative path | Source Git blob SHA-1 |
|---|---|
| `train_diffusion.py` | `a84255dfaffc0101fc603d3ce291ccc5acf20566` |
| `evaluate_diffusion.py` | `4b2842c12cca1f185943882640c12dfba3e01769` |
| `models/scerso_diffusion.py` | `db4e666929b788081bbe6ffdce8e58f63f9deff2` |
| `models/diffusion_core.py` | `cc330673e689f1336138848ddbbd62e75a52e464` |
| `utils/data_processor.py` | `7cae131371588765deca88d4137747cfeb3cb64c` |
| `utils/diffusion_schedule.py` | `6b2e5ddd9b042018e8cc92d2c2ebbf94976f5928` |
| `utils/emb_loader.py` | `883d143b041bdd2ace1be98799f1f696f35c2bd5` |

Every file's uploaded Git blob SHA-1 was compared against the corresponding local archive blob SHA-1 and matched.

## Scope and limitations

The accompanying English console adapter translates display-only legacy messages and checks reviewer-facing logs. A few comments and original diagnostic strings in the archived Python source are in Chinese; those original source bytes are intentionally preserved so the code remains verifiably identical to the locally tested version. All instructions and launcher output intended for reviewers are in English.

The **dataset** (`adamson_smoke.h5ad`, 22,344,537 bytes / approximately 21.31 MiB) is available in this public GitHub directory. The **lightweight EMA checkpoint** is available as a [versioned release asset](https://github.com/simonzqw/BaiZe/releases/download/adamson-smoke-v1/adamson_inference_ema.pth) (91,528,898 bytes; SHA-256 `3dad6cde7a2530dd4d2736e352b0f2f291ef7cf5e9617bcf0e3380eaa2d2ef89`). The `download_checkpoint.sh` helper installs it to `checkpoints/adamson_inference_ema.pth` and verifies its digest. The uploaded dataset's parent-source attribution and redistribution rights still require verification. A fresh-clone GPU evaluation has not yet been independently verified; see `PUBLICATION_STATUS.md`.

The demonstrated 3-epoch results are smoke-test results only; they are not the full manuscript Figure 2b benchmark.
