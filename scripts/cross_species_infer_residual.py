import argparse
import os

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

from models.cross_species_residual import CrossSpeciesResidualPredictor


def dmean(x):
    return np.asarray(x.mean(axis=0)).ravel() if issparse(x) else np.asarray(x).mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combined_h5ad', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--perturbations', nargs='+', required=True)
    p.add_argument('--perturb_col', default='perturbation')
    p.add_argument('--context_col', default='cell_context')
    p.add_argument('--control_key', default='control')
    p.add_argument('--species_col', default='species')
    p.add_argument('--human_value', default='human')
    p.add_argument('--atac_key', default='atac_feat')
    p.add_argument('--contextwise', action='store_true')
    p.add_argument('--aggregate_contexts', action='store_true')
    p.add_argument('--alpha_json', default=None)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location='cpu')
    vocab = ck['vocab']
    p2i = {p: i for i, p in enumerate(vocab)}

    comb = sc.read_h5ad(args.combined_h5ad)
    mouse = sc.read_h5ad(args.mouse_h5ad)
    hum = comb[comb.obs[args.species_col].astype(str).values == args.human_value]
    hctrl = dmean(hum[hum.obs[args.perturb_col].astype(str).values == args.control_key].X)
    src = {g: dmean(hum[hum.obs[args.perturb_col].astype(str).values == g].X) - hctrl for g in args.perturbations}

    model_args = ck.get('model_args', {})
    model = CrossSpeciesResidualPredictor(
        n_genes=ck.get('n_genes', mouse.n_vars),
        n_perturbations=len(vocab),
        atac_dim=ck.get('atac_dim', 256),
        use_atac=model_args.get('use_atac', args.atac_key in mouse.obsm),
        rank=model_args.get('rank', 64),
        hidden_dim=model_args.get('hidden_dim', 512),
        residual_scale=model_args.get('residual_scale', 0.01),
        learn_perturb_alpha=model_args.get('learn_perturb_alpha', True),
        alpha_init=model_args.get('alpha_init', -1.0),
        alpha_min=model_args.get('alpha_min', -3.0),
        alpha_max=model_args.get('alpha_max', 0.5),
    )
    model.load_state_dict(ck['model'])
    model.eval()

    alpha_map = {}
    if args.alpha_json and os.path.exists(args.alpha_json):
        import json
        with open(args.alpha_json) as f:
            alpha_map = json.load(f)

    preds = {}
    ctx_vals = sorted(set(mouse.obs[args.context_col].astype(str))) if args.contextwise else ['__global__']
    for g in args.perturbations:
        ps, ws = [], []
        for ctx in ctx_vals:
            mctx = mouse if ctx == '__global__' else mouse[mouse.obs[args.context_col].astype(str).values == ctx]
            ctrl = mctx[mctx.obs[args.perturb_col].astype(str).values == args.control_key]
            if ctrl.n_obs == 0:
                continue
            c = torch.tensor(dmean(ctrl.X)).float().unsqueeze(0)
            s = torch.tensor(src[g]).float().unsqueeze(0)
            a = torch.tensor(np.asarray(ctrl.obsm[args.atac_key]).mean(0)).float().unsqueeze(0) if args.atac_key in ctrl.obsm else None
            pid = torch.tensor([p2i.get(g, 0)]).long()
            with torch.no_grad():
                out = model(c, s, pid, a, alpha_override=alpha_map.get(g))
            pred = out['pred'].squeeze(0).numpy()
            key = f'{g}|{ctx}' if ctx != '__global__' else g
            preds[key] = pred
            ps.append(pred)
            ws.append(max(1, mctx.n_obs))
        if args.aggregate_contexts and ps:
            preds[g] = np.average(np.stack(ps), axis=0, weights=np.asarray(ws))

    np.savez_compressed(os.path.join(args.out_dir, 'mouse_cross_species_preds.npz'), **preds)
    np.savez_compressed(os.path.join(args.out_dir, 'predictions.npz'), **preds)


if __name__ == '__main__':
    main()
