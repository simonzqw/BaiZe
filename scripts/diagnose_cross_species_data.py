import argparse
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def normalize_pert(arr):
    x = np.asarray(arr).astype(str)
    x[np.isin(x, ['CTRL', 'CTRL1', 'ctrl', 'Control', 'vehicle', 'Vehicle'])] = 'control'
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--human_h5ad', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--perturb_col', default='perturbation')
    p.add_argument('--context_col', default='cell_context')
    p.add_argument('--perturb_vocab_path', default=None)
    p.add_argument('--out_json', default='cross_species_qc.json')
    args = p.parse_args()

    h = sc.read_h5ad(args.human_h5ad)
    m = sc.read_h5ad(args.mouse_h5ad)
    hp = normalize_pert(h.obs[args.perturb_col].values)
    mp = normalize_pert(m.obs[args.perturb_col].values)
    hs, ms = set(hp), set(mp)

    out = {
        'human_cells': int(h.n_obs),
        'mouse_cells': int(m.n_obs),
        'human_genes': int(h.n_vars),
        'mouse_genes': int(m.n_vars),
        'shared_var_names': int(len(set(h.var_names) & set(m.var_names))),
        'human_perturbations': int(len(hs)),
        'mouse_perturbations': int(len(ms)),
        'overlap_perturbation_count': int(len(hs & ms)),
        'overlap_perturbations': sorted(list(hs & ms)),
        'mouse_only_perturbations': sorted(list(ms - hs)),
        'human_control_cells': int(np.sum(hp == 'control')),
        'mouse_control_cells': int(np.sum(mp == 'control')),
        'human_X_is_sparse': bool(issparse(h.X)),
        'mouse_X_is_sparse': bool(issparse(m.X)),
    }

    if args.context_col in h.obs:
        hctx = h.obs[args.context_col].astype(str).values
        out['human_context_with_control'] = int(len(set(hctx[hp == 'control'])))
    if args.context_col in m.obs:
        mctx = m.obs[args.context_col].astype(str).values
        out['mouse_context_with_control'] = int(len(set(mctx[mp == 'control'])))

    if 'atac_feat' in h.obsm:
        arr = np.asarray(h.obsm['atac_feat'])
        out['human_atac_zero_ratio'] = float(np.mean(arr == 0))
        out['human_atac_mean_var'] = float(np.mean(np.var(arr, axis=0)))
    if 'atac_feat' in m.obsm:
        arr = np.asarray(m.obsm['atac_feat'])
        out['mouse_atac_zero_ratio'] = float(np.mean(arr == 0))
        out['mouse_atac_mean_var'] = float(np.mean(np.var(arr, axis=0)))

    if args.perturb_vocab_path:
        with open(args.perturb_vocab_path, 'r', encoding='utf-8') as f:
            vocab = set([ln.strip() for ln in f if ln.strip()])
        out['mouse_only_not_in_vocab'] = sorted(list((ms - hs) - vocab))

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
