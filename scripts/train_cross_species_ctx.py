import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor
from utils.diffusion_schedule import LossSecondMomentResampler, UniformTimestepSampler
from utils.emb_loader import GeneEmbeddingLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train cross-species diffusion model without cell_line token.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--pretrained_emb", type=str, default=None)
    p.add_argument("--split_strategy", type=str, default="perturbation", choices=["random", "perturbation"])
    p.add_argument("--split_col", type=str, default="split")
    p.add_argument("--perturb_parse_mode", type=str, default="raw", choices=["raw", "single_gene_suffix_clean", "double_gene_parse"])
    p.add_argument("--task_mode", type=str, default="single_gene", choices=["single_gene", "translation"])
    p.add_argument("--context_key", type=str, default="cell_context")
    p.add_argument("--perturb_vocab_path", type=str, default=None)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--perturb_dim", type=int, default=200)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--target_mode", type=str, default="delta", choices=["target", "delta"])
    p.add_argument("--dose_dim", type=int, default=32)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--cond_dropout", type=float, default=0.1)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--timestep_sampler", type=str, default="loss-second-moment", choices=["uniform", "loss-second-moment"])
    p.add_argument("--atac_key", type=str, default=None)
    p.add_argument("--atac_bank_path", type=str, default=None)
    p.add_argument("--background_key", type=str, default="cell_context")
    p.add_argument("--control_match_mode", type=str, default="atac_knn", choices=["random", "atac_knn"])
    p.add_argument("--control_match_k", type=int, default=16)
    p.add_argument("--control_match_scope", type=str, default="global", choices=["global", "cell_line"])
    p.add_argument("--control_prototype_mode", type=str, default="topk_weighted", choices=["single", "topk_mean", "topk_weighted"])
    p.add_argument("--lambda_topde", type=float, default=0.5)
    p.add_argument("--lambda_delta_corr", type=float, default=0.2)
    p.add_argument("--lambda_centroid", type=float, default=0.2)
    p.add_argument("--mean_loss_weight", type=float, default=10.0)
    p.add_argument("--diff_loss_weight", type=float, default=0.1)
    p.add_argument("--scgpt_gene_emb_path", type=str, default=None)
    p.add_argument("--gene_prior_scale", type=float, default=0.1)
    return p.parse_args()


def run_epoch(model, loader, optimizer, scaler, device, timestep_sampler, sample_steps, guidance_scale, drug_embeddings=None, train=True, args=None):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    metric_mses = []
    for batch in loader:
        ctrl = batch["rna_control"].to(device)
        target = batch["rna_target"].to(device)
        perturb = batch["perturb"].to(device)
        perturb_gene_idx = batch.get("perturb_gene_idx")
        is_control = batch.get("is_control")
        condition_id = batch.get("condition_id")
        source_flag = batch.get("source_flag")
        if perturb_gene_idx is not None:
            perturb_gene_idx = perturb_gene_idx.to(device)
        if is_control is not None:
            is_control = is_control.to(device)
        if condition_id is not None:
            condition_id = condition_id.to(device)
        if source_flag is not None:
            source_flag = source_flag.to(device)
        dose = batch["dose"].to(device) if "dose" in batch else None
        atac_feat = batch["atac_feat"].to(device) if "atac_feat" in batch else None
        drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

        with torch.set_grad_enabled(train):
            if train:
                t, weights = timestep_sampler.sample(ctrl.shape[0], device=device)
            else:
                t, weights = None, None
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = model(
                    rna_control=ctrl,
                    perturb=perturb,
                    target_rna=target,
                    dose=dose,
                    atac_feat=atac_feat,
                    drug_feat=drug_feat,
                    t=t,
                    weights=weights,
                    perturb_gene_idx=perturb_gene_idx,
                    is_control=is_control,
                    condition_id=condition_id,
                    source_flag=source_flag,
                    mean_loss_weight=args.mean_loss_weight,
                    diff_loss_weight=args.diff_loss_weight,
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                timestep_sampler.update_with_losses(t, torch.full_like(t, float(loss.detach().item()), dtype=torch.float32))
            else:
                pred = model.predict_single(
                    rna_control=ctrl,
                    perturb=perturb,
                    dose=dose,
                    atac_feat=atac_feat,
                    drug_feat=drug_feat,
                    sample_steps=sample_steps,
                    guidance_scale=guidance_scale,
                    perturb_gene_idx=perturb_gene_idx,
                    is_control=is_control,
                    condition_id=condition_id,
                    source_flag=source_flag,
                )
                metric_mses.append(float(torch.mean((pred - target) ** 2).detach().item()))

        losses.append(float(loss.detach().item()))
    loss_mean = float(np.mean(losses)) if len(losses) > 0 else 0.0
    metric_mse = float(np.mean(metric_mses)) if len(metric_mses) > 0 else 0.0
    return loss_mean, metric_mse


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy,
        split_col=args.split_col,
        perturb_parse_mode=args.perturb_parse_mode,
        task_mode=args.task_mode,
        perturb_vocab_path=args.perturb_vocab_path,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.context_key,
    )
    n_genes, n_perts, _ = processor.load_data()
    train_loader, val_loader, _ = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.context_key,
        control_match_mode=args.control_match_mode,
        control_match_k=args.control_match_k,
        control_match_scope=args.control_match_scope,
        control_prototype_mode=args.control_prototype_mode,
    )

    pretrained_weights = None
    pretrained_gene_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()
        if getattr(processor, "idx_to_perturb_gene", None):
            gene_loader = GeneEmbeddingLoader(args.pretrained_emb, processor.idx_to_perturb_gene)
            pretrained_gene_weights = gene_loader.load_weights()

    atac_dim = processor.atac_dim if getattr(processor, "atac_features", None) is not None else 0
    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        pretrained_weights=pretrained_weights,
        pretrained_gene_weights=pretrained_gene_weights,
        perturb_dim=args.perturb_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        timesteps=args.timesteps,
        target_mode=args.target_mode,
        dose_dim=args.dose_dim,
        time_dim=args.time_dim,
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=atac_dim,
        cond_dropout=args.cond_dropout,
        n_perturb_genes=len(getattr(processor, "perturb_gene_vocab", []) or []),
        task_mode=args.task_mode,
        n_conditions=getattr(processor, "n_conditions", 0),
    ).to(device)

    if args.timestep_sampler == "loss-second-moment":
        timestep_sampler = LossSecondMomentResampler(args.timesteps)
    else:
        timestep_sampler = UniformTimestepSampler(args.timesteps)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    best_val_loss = float("inf")
    best_val_pred_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            timestep_sampler=timestep_sampler,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            drug_embeddings=drug_embeddings,
            train=True,
            args=args,
        )
        val_loss, val_pred_mse = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            timestep_sampler=timestep_sampler,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            drug_embeddings=drug_embeddings,
            train=False,
            args=args,
        )
        print(
            f"[E{epoch:03d}/{args.epochs:03d}] "
            f"train_diff_loss={train_loss:.6f} val_diff_loss={val_loss:.6f} val_pred_mse={val_pred_mse:.6f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": args,
            "n_genes": n_genes,
            "n_perts": n_perts,
            "perturb_categories": processor.perturb_categories,
            "atac_dim": atac_dim,
            "use_atac": bool(processor.atac_features is not None),
            "best_val_loss": best_val_loss,
            "best_val_pred_mse": best_val_pred_mse,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_ctx_diff.pth"))
            print(f"  ↳ best diffusion loss updated: val_diff_loss={best_val_loss:.6f}")
        if val_pred_mse < best_val_pred_mse:
            best_val_pred_mse = val_pred_mse
            ckpt["best_val_pred_mse"] = best_val_pred_mse
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_ctx_diff_predmse.pth"))
            print(f"  ↳ best prediction mse updated: val_pred_mse={best_val_pred_mse:.6f}")

    print(f">>> done. diffusion checkpoints in {args.save_dir}")


if __name__ == "__main__":
    main()
