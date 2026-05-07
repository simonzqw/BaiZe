import argparse
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def safe_pearson(x, y):
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--pred_npz', required=True)
    p.add_argument('--control_key', default='control')
    p.add_argument('--perturb_col', default='perturbation')
    p.add_argument('--context_col', default='cell_context')
    p.add_argument('--output_json', default='cross_species_mouse_eval.json')
    args = p.parse_args()

    adata = sc.read_h5ad(args.mouse_h5ad)
    pred = np.load(args.pred_npz, allow_pickle=True)
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    pert = adata.obs[args.perturb_col].astype(str).values
    pert[np.isin(pert, ['CTRL', 'CTRL1', 'ctrl', 'Control', 'vehicle', 'Vehicle'])] = args.control_key
    ctx = adata.obs[args.context_col].astype(str).values if args.context_col in adata.obs else np.array(['global'] * adata.n_obs)

    out = {'eval_mode': 'global_control_eval'}
    for k in pred.files:
        idx = np.where(pert == k)[0]
        if len(idx) == 0:
            continue
        cands = np.where(pert == args.control_key)[0]
        if len(cands) == 0:
            continue
        ctrl = X[cands].mean(axis=0)
        t = X[idx].mean(axis=0)
        arr = np.asarray(pred[k])
        if arr.ndim == 1:
            pvec = arr
        elif arr.ndim == 2:
            pvec = arr.mean(axis=0)
        else:
            raise ValueError(f'Unexpected prediction shape for {k}: {arr.shape}')
        td = t - ctrl
        pd = pvec - ctrl
        top = np.argsort(np.abs(td))[-min(20, len(td)):]
        out[k] = {
            'pearson_all': safe_pearson(pvec, t),
            'pearson_delta': safe_pearson(pd, td),
            'top20_delta_pearson': safe_pearson(pd[top], td[top]),
            'top20_delta_mse': float(np.mean((pd[top] - td[top]) ** 2)),
            'opposite_direction_top20': float(np.mean(np.sign(pd[top]) != np.sign(td[top]))),
            'n_contexts': int(len(set(ctx[idx]))),
        }
    vals = [v for v in out.values() if isinstance(v, dict) and 'top20_delta_pearson' in v]
    if len(vals) > 0:
        out['summary'] = {
            'mean_top20_delta_pearson': float(np.mean([x['top20_delta_pearson'] for x in vals])),
            'mean_top20_delta_mse': float(np.mean([x['top20_delta_mse'] for x in vals])),
            'mean_opposite_direction_top20': float(np.mean([x['opposite_direction_top20'] for x in vals])),
            'mean_pearson_delta': float(np.mean([x['pearson_delta'] for x in vals])),
        }
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
