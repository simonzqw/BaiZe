import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


def dense_mean(x):
    return np.asarray(x.mean(axis=0)).ravel() if issparse(x) else np.asarray(x).mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combined_h5ad', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--perturb_col', default='perturbation')
    p.add_argument('--context_col', default='cell_context')
    p.add_argument('--species_col', default='species')
    p.add_argument('--split_col', default='split')
    p.add_argument('--control_key', default='control')
    p.add_argument('--atac_key', default='atac_feat')
    p.add_argument('--perturbations', nargs='+', default=['ARID1A', 'PDCD1'])
    args = p.parse_args()

    adata = sc.read_h5ad(args.combined_h5ad)
    mouse = sc.read_h5ad(args.mouse_h5ad)
    print('gene_order_equal:', np.array_equal(adata.var_names.values, mouse.var_names.values))
    print('combined shape:', adata.shape, 'mouse shape:', mouse.shape)
    print('species counts\n', adata.obs[args.species_col].value_counts())
    print('split counts\n', adata.obs[args.split_col].value_counts())
    print('mouse perturb counts\n', mouse.obs[args.perturb_col].value_counts())
    print('mouse context counts\n', mouse.obs[args.context_col].value_counts())
    if args.atac_key in adata.obsm:
        print('combined atac', adata.obsm[args.atac_key].shape)
    if args.atac_key in mouse.obsm:
        print('mouse atac', mouse.obsm[args.atac_key].shape)

    human = adata[adata.obs[args.species_col].astype(str).values == 'human']
    print(pd.crosstab(human.obs[args.context_col], human.obs[args.perturb_col]))
    print(pd.crosstab(mouse.obs[args.context_col], mouse.obs[args.perturb_col]))

    hobs, mobs = human.obs, mouse.obs
    h_ctrl = human[hobs[args.perturb_col].astype(str).values == args.control_key]
    m_ctrl = mouse[mobs[args.perturb_col].astype(str).values == args.control_key]
    h_ctrl_mean, m_ctrl_mean = dense_mean(h_ctrl.X), dense_mean(m_ctrl.X)

    for g in args.perturbations:
        h_g = human[hobs[args.perturb_col].astype(str).values == g]
        m_g = mouse[mobs[args.perturb_col].astype(str).values == g]
        if h_g.n_obs == 0 or m_g.n_obs == 0:
            print(g, 'missing in human or mouse')
            continue
        h_delta = dense_mean(h_g.X) - h_ctrl_mean
        m_delta = dense_mean(m_g.X) - m_ctrl_mean
        cos = float(np.dot(h_delta, m_delta) / (np.linalg.norm(h_delta) * np.linalg.norm(m_delta) + 1e-8))
        print(g, 'human_n', h_g.n_obs, 'mouse_n', m_g.n_obs, 'human_norm', float(np.linalg.norm(h_delta)), 'mouse_norm', float(np.linalg.norm(m_delta)), 'cos', cos)


if __name__ == '__main__':
    main()
