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

The **dataset** (`adamson_smoke.h5ad`, 22,344,537 bytes / approximately 21.31 MiB) is now available in this public GitHub directory. The **lightweight EMA checkpoint** (`checkpoints/adamson_inference_ema.pth`, approximately 87.29 MiB) is not yet publicly hosted. The uploaded dataset's parent-source attribution and redistribution rights still require verification. Check `PUBLICATION_STATUS.md` before claiming an independently executable public reproduction.

The demonstrated 3-epoch results are smoke-test results only; they are not the full manuscript Figure 2b benchmark.
