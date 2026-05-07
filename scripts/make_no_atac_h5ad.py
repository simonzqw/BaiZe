import argparse
import scanpy as sc

p = argparse.ArgumentParser()
p.add_argument('--in_h5ad', required=True)
p.add_argument('--out_h5ad', required=True)
args = p.parse_args()
adata = sc.read_h5ad(args.in_h5ad)
if 'atac_feat' in adata.obsm:
    del adata.obsm['atac_feat']
for c in ['atac_replicate_id', 'atac_context_type', 'atac_is_condition_level']:
    if c in adata.obs:
        del adata.obs[c]
for k in ['atac_dim', 'atac_note', 'atac_projection', 'atac_replicates_used', 'atac_source_npz']:
    if k in adata.uns:
        del adata.uns[k]
adata.write_h5ad(args.out_h5ad)
