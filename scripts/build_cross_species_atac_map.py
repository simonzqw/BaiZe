import argparse
import gzip
import os
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import pyBigWig
except Exception:
    pyBigWig = None


def parse_gtf_gene_tss(gtf_path: str, target_symbols: Iterable[str], promoter_bp: int) -> pd.DataFrame:
    target = set(str(x) for x in target_symbols)
    rows = []
    opener = gzip.open if gtf_path.endswith('.gz') else open
    with opener(gtf_path, 'rt', encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('#'):
                continue
            parts = ln.rstrip('\n').split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom, _, _, start, end, _, strand, _, attrs = parts
            attr_map = {k: v for k, v in re.findall(r'(\S+) "([^"]+)"', attrs)}
            gene_symbol = attr_map.get('gene_name', attr_map.get('gene_id', ''))
            if gene_symbol not in target:
                continue
            gene_id = attr_map.get('gene_id', '')
            gene_type = attr_map.get('gene_type', attr_map.get('gene_biotype', ''))
            start_i, end_i = int(start), int(end)
            tss_1based = start_i if strand == '+' else end_i
            s0 = max(0, tss_1based - promoter_bp - 1)
            e0 = tss_1based + promoter_bp
            rows.append((chrom, s0, e0, gene_symbol, strand, gene_id, gene_type, tss_1based))
    return pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'gene_symbol', 'strand', 'gene_id', 'gene_type', 'tss_1based'])


def compute_bigwig_promoter_signal(bw_path: str, promoter_df: pd.DataFrame, gene_to_idx: Dict[str, int], n_genes: int) -> np.ndarray:
    vec = np.zeros((n_genes,), dtype=np.float32)
    if pyBigWig is None:
        return vec
    bw = pyBigWig.open(bw_path)
    try:
        for _, r in promoter_df.iterrows():
            gid = gene_to_idx.get(str(r['gene_symbol']))
            if gid is None:
                continue
            try:
                v = bw.stats(str(r['chrom']), int(r['start']), int(r['end']), type='mean')[0]
            except RuntimeError:
                v = 0.0
            if v is None or np.isnan(v):
                v = 0.0
            vec[gid] = float(v)
    finally:
        bw.close()
    return vec


def zscore_log1p(vec: np.ndarray) -> np.ndarray:
    x = np.log1p(np.clip(vec, 0, None)).astype(np.float32)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description='Build cross-species promoter ATAC map in shared ortholog space.')
    p.add_argument('--shared_gene_order', required=True)
    p.add_argument('--human_gtf', required=True)
    p.add_argument('--mouse_gtf', required=True)
    p.add_argument('--human_bw_list', required=True)
    p.add_argument('--mouse_bw_list', required=True)
    p.add_argument('--promoter_bp', type=int, default=2000)
    p.add_argument('--out_npz', required=True)
    p.add_argument('--out_tsv', required=True)
    args = p.parse_args()

    shared = pd.read_csv(args.shared_gene_order, sep='\t')
    hcol = 'human_gene_symbol'
    mcol = 'mouse_gene_symbol'
    if hcol not in shared.columns or mcol not in shared.columns:
        raise ValueError('shared_gene_order must include human_gene_symbol and mouse_gene_symbol')

    human_symbols = shared[hcol].astype(str).tolist()
    mouse_symbols = shared[mcol].astype(str).tolist()
    n = len(shared)
    h_map = {g: i for i, g in enumerate(human_symbols)}
    m_map = {g: i for i, g in enumerate(mouse_symbols)}

    human_prom = parse_gtf_gene_tss(args.human_gtf, human_symbols, args.promoter_bp)
    mouse_prom = parse_gtf_gene_tss(args.mouse_gtf, mouse_symbols, args.promoter_bp)

    human_bw = pd.read_csv(args.human_bw_list, sep='\t')
    mouse_bw = pd.read_csv(args.mouse_bw_list, sep='\t')
    for df in (human_bw, mouse_bw):
        if 'sample_id' not in df.columns or 'bw_path' not in df.columns:
            raise ValueError('bw list tsv must include sample_id and bw_path columns')

    out = {'genes': np.array(human_symbols, dtype=object), 'human_gene_symbol': np.array(human_symbols, dtype=object), 'mouse_gene_symbol': np.array(mouse_symbols, dtype=object)}
    summary = []

    for _, row in human_bw.iterrows():
        sid = str(row['sample_id'])
        vec = compute_bigwig_promoter_signal(str(row['bw_path']), human_prom, h_map, n)
        out[f'{sid}_raw'] = vec
        out[f'{sid}_log1p_z'] = zscore_log1p(vec)
        summary.append({'sample_id': sid, 'species': 'human', 'nonzero': int((vec > 0).sum())})

    for _, row in mouse_bw.iterrows():
        sid = str(row['sample_id'])
        vec = compute_bigwig_promoter_signal(str(row['bw_path']), mouse_prom, m_map, n)
        out[f'{sid}_raw'] = vec
        out[f'{sid}_log1p_z'] = zscore_log1p(vec)
        summary.append({'sample_id': sid, 'species': 'mouse', 'nonzero': int((vec > 0).sum())})

    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)
    np.savez_compressed(args.out_npz, **out)
    pd.DataFrame(summary).to_csv(args.out_tsv, sep='\t', index=False)


if __name__ == '__main__':
    main()
