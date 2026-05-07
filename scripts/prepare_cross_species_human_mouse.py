import argparse
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--human_h5ad', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--shared_gene_order_tsv', required=True)
    p.add_argument('--out_h5ad', required=True)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    h = sc.read_h5ad(args.human_h5ad)
    m = sc.read_h5ad(args.mouse_h5ad)
    s = pd.read_csv(args.shared_gene_order_tsv, sep='\t')

    pairs = []
    for _, row in s.iterrows():
        hg = str(row['human_gene_symbol'])
        mg = str(row['mouse_gene_symbol'])
        if hg in h.var_names and mg in m.var_names:
            pairs.append((hg, mg, h.var_names.get_loc(hg), m.var_names.get_loc(mg)))

    human_genes = [x[0] for x in pairs]
    hi = [x[2] for x in pairs]
    mi = [x[3] for x in pairs]

    h2 = h[:, hi].copy()
    m2 = m[:, mi].copy()
    h2.var_names = human_genes
    m2.var_names = human_genes

    h2.obs['species'] = 'human'
    m2.obs['species'] = 'mouse'
    h2.obs['dataset'] = 'GSE119450'
    m2.obs['dataset'] = 'GSE203592'

    rng = np.random.RandomState(args.seed)
    hsplit = np.array(['train'] * h2.n_obs, dtype=object)
    nval = max(1, int(h2.n_obs * args.val_frac))
    vidx = rng.choice(np.arange(h2.n_obs), size=nval, replace=False)
    hsplit[vidx] = 'val'
    h2.obs['split'] = hsplit
    m2.obs['split'] = 'test'

    combined = sc.concat([h2, m2], join='inner')
    combined.write_h5ad(args.out_h5ad)


if __name__ == '__main__':
    main()
