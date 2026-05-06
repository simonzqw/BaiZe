import argparse
import os
import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ortholog_tsv', required=True)
    p.add_argument('--shared_gene_order', required=True)
    p.add_argument('--human_gtf', required=True)
    p.add_argument('--mouse_gtf', required=True)
    p.add_argument('--human_bw_dir', required=True)
    p.add_argument('--mouse_bw_dir', required=True)
    p.add_argument('--promoter_bp', type=int, default=2000)
    p.add_argument('--out_npz', required=True)
    return p.parse_args()


def main():
    args = get_args()
    shared = pd.read_csv(args.shared_gene_order, sep='\t')
    n = len(shared)
    out_dir = os.path.dirname(args.out_npz) or '.'
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out_npz, genes=shared.iloc[:, 0].astype(str).values, human_promoter_atac=np.zeros(n, dtype=np.float32), mouse_promoter_atac=np.zeros(n, dtype=np.float32))
    pd.DataFrame({'metric': ['n_shared_genes', 'promoter_bp'], 'value': [n, args.promoter_bp]}).to_csv(os.path.splitext(args.out_npz)[0] + '_summary.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
