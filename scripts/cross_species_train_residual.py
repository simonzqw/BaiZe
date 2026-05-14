import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.cross_species_losses import delta_norm_loss, residual_l2_loss, sign_consistency_loss, topk_corr_loss, weighted_delta_mse
from models.cross_species_residual import CrossSpeciesResidualPredictor


def calc_loss(o, td, gw, args):
    return (
        args.delta_mse_weight * weighted_delta_mse(o['pred_delta'], td, gw)
        + args.topk_corr_weight * topk_corr_loss(o['pred_delta'], td)
        + args.sign_weight * sign_consistency_loss(o['pred_delta'], td)
        + args.norm_weight * delta_norm_loss(o['pred_delta'], td)
        + args.correction_l2_weight * residual_l2_loss(o['correction'])
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pseudobulk_npz', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--perturb_vocab_path', required=True)
    p.add_argument('--use_atac', action='store_true')
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--residual_scale', type=float, default=0.01)
    p.add_argument('--scgpt_gene_basis', default=None)
    p.add_argument('--freeze_gene_basis', action='store_true')
    p.add_argument('--learn_perturb_alpha', action='store_true')
    p.add_argument('--alpha_init', type=float, default=-1.0)
    p.add_argument('--alpha_min', type=float, default=-3.0)
    p.add_argument('--alpha_max', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--delta_mse_weight', type=float, default=1.0)
    p.add_argument('--topk_corr_weight', type=float, default=0.5)
    p.add_argument('--sign_weight', type=float, default=0.2)
    p.add_argument('--norm_weight', type=float, default=0.1)
    p.add_argument('--correction_l2_weight', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--grad_clip', type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = np.load(args.pseudobulk_npz, allow_pickle=True)
    with open(args.perturb_vocab_path) as f:
        vocab = [x.strip() for x in f if x.strip()]
    p2i = {p: i for i, p in enumerate(vocab)}
    pid = np.array([p2i.get(x, 0) for x in d['perturbation'].astype(str)], dtype=np.int64)
    gene_weight = torch.tensor(d['gene_weight']).float() if 'gene_weight' in d else torch.ones_like(torch.tensor(d['true_delta']).float())

    ds = TensorDataset(
        torch.tensor(d['control_mean']).float(),
        torch.tensor(d['source_delta']).float(),
        torch.tensor(d['atac_mean']).float(),
        torch.tensor(pid).long(),
        torch.tensor(d['target_mean']).float(),
        torch.tensor(d['true_delta']).float(),
        gene_weight,
    )
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)


    gene_basis_init = None
    if args.scgpt_gene_basis is not None:
        gene_basis_init = np.load(args.scgpt_gene_basis).astype(np.float32)
        print('loaded scGPT gene basis:', args.scgpt_gene_basis, gene_basis_init.shape)
        expected_shape = (args.rank, ds.tensors[0].shape[1])
        if tuple(gene_basis_init.shape) != expected_shape:
            raise ValueError(
                f'scGPT gene basis shape {gene_basis_init.shape} does not match '
                f'expected shape {expected_shape}. '
                f'Please make sure --rank matches the basis rank and gene order matches the h5ad.'
            )

    model = CrossSpeciesResidualPredictor(
        n_genes=ds.tensors[0].shape[1],
        n_perturbations=len(vocab),
        atac_dim=ds.tensors[2].shape[1],
        use_atac=args.use_atac,
        rank=args.rank,
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        learn_perturb_alpha=args.learn_perturb_alpha,
        alpha_init=args.alpha_init,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        gene_basis_init=gene_basis_init,
        freeze_gene_basis=args.freeze_gene_basis,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == 'cuda')

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for c, s, a, p_id, _t, td, gw in train_dl:
            c, s, a, p_id, td, gw = c.to(device), s.to(device), a.to(device), p_id.to(device), td.to(device), gw.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == 'cuda'):
                o = model(c, s, p_id, a if args.use_atac else None)
                loss = calc_loss(o, td, gw, args)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for c, s, a, p_id, _t, td, gw in val_dl:
                c, s, a, p_id, td, gw = c.to(device), s.to(device), a.to(device), p_id.to(device), td.to(device), gw.to(device)
                o = model(c, s, p_id, a if args.use_atac else None)
                v = calc_loss(o, td, gw, args)
                vtot += v.item() * c.shape[0]
                vn += c.shape[0]
        val_avg = vtot / max(vn, 1)
        if val_avg < best:
            best = val_avg
            torch.save({
                'model': model.state_dict(),
                'vocab': vocab,
                'n_genes': ds.tensors[0].shape[1],
                'atac_dim': ds.tensors[2].shape[1],
                'model_args': {
                    'use_atac': args.use_atac,
                    'rank': args.rank,
                    'hidden_dim': args.hidden_dim,
                    'residual_scale': args.residual_scale,
                    'alpha_min': args.alpha_min,
                    'alpha_max': args.alpha_max,
                    'alpha_init': args.alpha_init,
                    'learn_perturb_alpha': args.learn_perturb_alpha,
                    'scgpt_gene_basis': args.scgpt_gene_basis,
                    'freeze_gene_basis': args.freeze_gene_basis,
                },
                'train_args': vars(args),
                'best_loss': best,
            }, os.path.join(args.save_dir, 'best_model_cross_species_residual.pth'))
        if ep % 10 == 0:
            print(f'epoch={ep} val_loss={val_avg:.6f} best={best:.6f}')


if __name__ == '__main__':
    main()
