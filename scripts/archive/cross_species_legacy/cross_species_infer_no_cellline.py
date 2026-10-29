import argparse
import json
import numpy as np
import scanpy as sc
import torch

from models.scerso_diffusion import PerturbationDiffusionPredictor


def normalize_control_labels(arr, control_label='control'):
    aliases = {'CTRL', 'CTRL1', 'ctrl', 'Control', 'vehicle', 'Vehicle', control_label}
    x = np.asarray(arr).astype(str)
    x[np.isin(x, list(aliases))] = 'control'
    return x


def row_pearson(a, b):
    out = []
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(x, y)[0, 1]))
    return np.array(out, dtype=np.float32)


def topk_delta_metrics(pred, true, ctrl, ks=(20, 50, 200)):
    m = {}
    dp, dt = pred - ctrl, true - ctrl
    for k in ks:
        vals_p, vals_t = [], []
        for i in range(pred.shape[0]):
            kk = min(k, pred.shape[1])
            idx = np.argsort(np.abs(dt[i]))[-kk:]
            vals_p.append(dp[i, idx])
            vals_t.append(dt[i, idx])
        vp, vt = np.stack(vals_p), np.stack(vals_t)
        m[f'top{k}_delta_pearson'] = float(np.mean(row_pearson(vp, vt)))
        m[f'top{k}_delta_mse'] = float(np.mean((vp - vt) ** 2))
    return m


def build_control_proto_global_mean(ctrl_pool):
    return np.mean(ctrl_pool, axis=0, keepdims=True)


def build_control_proto_random(ctrl_pool, n, rng):
    idx = rng.randint(0, ctrl_pool.shape[0], size=n)
    return ctrl_pool[idx]


def build_control_proto_atac_knn(ctrl_pool, ctrl_atac, q_atac, topk=16):
    out = []
    for i in range(q_atac.shape[0]):
        d = np.sum((ctrl_atac - q_atac[i:i+1]) ** 2, axis=1)
        idx = np.argsort(d)[:max(1, min(topk, len(d)))]
        out.append(np.mean(ctrl_pool[idx], axis=0))
    return np.stack(out, axis=0)


def build_model_from_checkpoint(ckpt, n_genes, device):
    args = ckpt.get('args', argparse.Namespace())
    n_perts = int(ckpt['model_state_dict']['perturb_embedding.weight'].shape[0])
    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        perturb_dim=getattr(args, 'perturb_dim', 200),
        hidden_dims=getattr(args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(args, 'dropout', 0.1),
        timesteps=getattr(args, 'timesteps', 1000),
        target_mode=getattr(args, 'target_mode', 'delta'),
        use_atac=getattr(args, 'atac_key', None) is not None,
        atac_dim=int(ckpt['model_state_dict'].get('atac_projection.0.weight', torch.zeros(512, 0)).shape[1]) if 'atac_projection.0.weight' in ckpt['model_state_dict'] else 0,
        n_perturb_genes=1,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--out_dir', default='cross_species_out')
    p.add_argument('--perturbations', nargs='+', default=['ARID1A', 'PDCD1'])
    p.add_argument('--atac_key', default='atac_feat')
    p.add_argument('--control_label', default='control')
    p.add_argument('--control_mode', default='atac_knn', choices=['atac_knn', 'random', 'global_mean'])
    p.add_argument('--max_cells_per_pert', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--control_topk', type=int, default=16)
    p.add_argument('--sample_steps', type=int, default=20)
    p.add_argument('--guidance_scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    adata = sc.read_h5ad(args.mouse_h5ad)
    pert = normalize_control_labels(adata.obs['perturbation'].astype(str).values, args.control_label)
    x = np.asarray(adata.X)
    ctrl_idx = np.where(pert == 'control')[0]
    ctrl_pool = x[ctrl_idx]
    ctrl_atac = np.asarray(adata.obsm[args.atac_key][ctrl_idx]) if args.atac_key in adata.obsm else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, adata.n_vars, device)
    use_atac = bool(getattr(model, 'use_atac', False) and ctrl_atac is not None)
    if not use_atac:
        print('RNA-only checkpoint: model will receive atac_feat=None')

    report = {}
    for p_name in args.perturbations:
        idx = np.where(pert == p_name)[0]
        if len(idx) == 0:
            continue
        if len(idx) > args.max_cells_per_pert:
            idx = rng.choice(idx, size=args.max_cells_per_pert, replace=False)

        y_true = x[idx]
        q_atac = np.asarray(adata.obsm[args.atac_key][idx]) if (use_atac and args.atac_key in adata.obsm) else None
        if args.control_mode == 'global_mean':
            ctrl_proto = np.repeat(build_control_proto_global_mean(ctrl_pool), repeats=len(idx), axis=0)
        elif args.control_mode == 'random':
            ctrl_proto = build_control_proto_random(ctrl_pool, len(idx), rng)
        else:
            ctrl_proto = build_control_proto_atac_knn(ctrl_pool, ctrl_atac, q_atac, topk=args.control_topk)

        preds = []
        for s in range(0, len(idx), args.batch_size):
            e = min(s + args.batch_size, len(idx))
            ctrl_t = torch.tensor(ctrl_proto[s:e], dtype=torch.float32, device=device)
            atac_t = torch.tensor(q_atac[s:e], dtype=torch.float32, device=device) if q_atac is not None else None
            with torch.no_grad():
                pr = model.sample(
                    rna_control=ctrl_t,
                    perturb=torch.zeros((e - s,), dtype=torch.long, device=device),
                    perturb_gene_idx=torch.ones((e - s,), dtype=torch.long, device=device),
                    is_control=torch.zeros((e - s,), dtype=torch.float32, device=device),
                    atac_feat=atac_t,
                    sample_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                )
            preds.append(pr.cpu().numpy())
        y_pred = np.concatenate(preds, axis=0)

        m = {
            'n_cells': int(len(idx)),
            'pearson': float(np.mean(row_pearson(y_pred, y_true))),
            'delta_pearson': float(np.mean(row_pearson(y_pred - ctrl_proto, y_true - ctrl_proto))),
            'mse': float(np.mean((y_pred - y_true) ** 2)),
        }
        m.update(topk_delta_metrics(y_pred, y_true, ctrl_proto))

        p_mean, t_mean, c_mean = y_pred.mean(axis=0, keepdims=True), y_true.mean(axis=0, keepdims=True), ctrl_proto.mean(axis=0, keepdims=True)
        pdm, tdm = p_mean - c_mean, t_mean - c_mean
        m['perturb_pearson'] = float(row_pearson(p_mean, t_mean)[0])
        m['perturb_delta_pearson'] = float(row_pearson(pdm, tdm)[0])
        for k in [20, 50, 200]:
            kk = min(k, tdm.shape[1])
            tidx = np.argsort(np.abs(tdm[0]))[-kk:]
            tp, tt = pdm[:, tidx], tdm[:, tidx]
            m[f'perturb_top{k}_delta_pearson'] = float(row_pearson(tp, tt)[0])
            m[f'perturb_top{k}_delta_mse'] = float(np.mean((tp - tt) ** 2))
            valid = np.abs(tt[0]) > 1e-8
            m[f'perturb_top{k}_direction_accuracy'] = float(1.0 - np.mean(np.sign(tp[0, valid]) != np.sign(tt[0, valid]))) if valid.sum() > 0 else 0.0
        report[p_name] = m

    with open(f"{args.out_dir}/metrics.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
