# Cross-species module (v2)

This module is independent from the diffusion mainline (`train_diffusion.py`, `evaluate_diffusion.py`, `predict_diffusion.py`).

Pipeline:
1. Diagnose data (`scripts/cross_species_diagnose_data.py`).
2. Build bootstrap pseudo-bulk (`scripts/cross_species_build_pseudobulk.py`).
3. Train residual model (`scripts/cross_species_train_residual.py`).
4. Run context-wise inference (`scripts/cross_species_infer_residual.py`).
5. Optional perturbation-specific calibration via `configs/cross_species/perturb_alpha.json`.
