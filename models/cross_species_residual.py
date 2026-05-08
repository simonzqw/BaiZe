import math

import torch
import torch.nn as nn


class CrossSpeciesResidualPredictor(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        atac_dim: int = 256,
        perturb_dim: int = 128,
        hidden_dim: int = 512,
        rank: int = 64,
        dropout: float = 0.1,
        residual_scale: float = 0.01,
        use_atac: bool = True,
        learn_perturb_alpha: bool = True,
        alpha_init: float = -1.0,
        alpha_min: float = -3.0,
        alpha_max: float = 0.5,
        gene_basis_init=None,
        freeze_gene_basis: bool = False,
    ) -> None:
        super().__init__()
        self.use_atac = use_atac
        self.residual_scale = residual_scale
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.freeze_gene_basis = freeze_gene_basis

        self.control_encoder = self._encoder(n_genes, hidden_dim, perturb_dim, dropout)
        self.source_delta_encoder = self._encoder(n_genes, hidden_dim, perturb_dim, dropout)
        self.perturb_emb = nn.Embedding(n_perturbations, perturb_dim)
        self.atac_encoder = self._encoder(atac_dim, hidden_dim, perturb_dim, dropout) if use_atac else None

        in_dim = perturb_dim * (4 if use_atac else 3)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.coeff_head = nn.Linear(hidden_dim, rank)

        if gene_basis_init is not None:
            if tuple(gene_basis_init.shape) != (rank, n_genes):
                raise ValueError(
                    f"gene_basis_init shape {tuple(gene_basis_init.shape)} does not match "
                    f"(rank, n_genes)=({rank}, {n_genes})"
                )
            if not torch.is_tensor(gene_basis_init):
                gene_basis_init = torch.tensor(gene_basis_init, dtype=torch.float32)
            else:
                gene_basis_init = gene_basis_init.float()
            self.gene_basis = nn.Parameter(gene_basis_init.clone())
        else:
            self.gene_basis = nn.Parameter(torch.randn(rank, n_genes) * 0.02)

        if freeze_gene_basis:
            self.gene_basis.requires_grad_(False)

        self.gate_head = nn.Sequential(nn.Linear(hidden_dim, n_genes), nn.Sigmoid())
        self.alpha_emb = nn.Embedding(n_perturbations, 1) if learn_perturb_alpha else None
        self.register_buffer("fixed_alpha", torch.tensor(float(alpha_init)))

        nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        if self.alpha_emb is not None:
            ratio = (alpha_init - alpha_min) / (alpha_max - alpha_min)
            ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
            raw_init = math.log(ratio / (1.0 - ratio))
            nn.init.constant_(self.alpha_emb.weight, raw_init)

    @staticmethod
    def _encoder(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _alpha(self, perturb_id: torch.Tensor) -> torch.Tensor:
        if self.alpha_emb is None:
            return self.fixed_alpha.view(1, 1).expand(perturb_id.shape[0], 1)
        raw = self.alpha_emb(perturb_id)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw)

    def forward(self, control, source_delta, perturb_id, atac_feat=None, alpha_override=None):
        zc = self.control_encoder(control)
        zs = self.source_delta_encoder(source_delta)
        zp = self.perturb_emb(perturb_id)
        parts = [zc, zs, zp]
        if self.use_atac:
            if atac_feat is None:
                raise ValueError("atac_feat is required when use_atac=True")
            parts.append(self.atac_encoder(atac_feat))
        h = self.fusion(torch.cat(parts, dim=-1))

        low_rank = self.coeff_head(h) @ self.gene_basis
        gate = self.gate_head(h)
        alpha = self._alpha(perturb_id)
        if alpha_override is not None:
            alpha = torch.full_like(alpha, float(alpha_override))

        correction = alpha * gate * (self.residual_scale * low_rank)
        pred_delta = source_delta + correction
        pred = control + pred_delta
        return {"pred": pred, "pred_delta": pred_delta, "source_delta": source_delta, "correction": correction, "alpha": alpha, "gate": gate}
