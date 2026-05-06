import argparse
import numpy as np
import scanpy as sc

p = argparse.ArgumentParser()
p.add_argument('--h5ad_in', required=True)
p.add_argument('--h5ad_out', required=True)
p.add_argument('--atac_npz', required=True)
p.add_argument('--species', required=True)
p.add_argument('--replicate_keys', nargs='+', required=True)
p.add_argument('--mode', default='round_robin', choices=['round_robin', 'random'])
p.add_argument('--project_dim', type=int, default=256)
p.add_argument('--seed', type=int, default=20260506)
args = p.parse_args()
adata = sc.read_h5ad(args.h5ad_in)
bank = np.load(args.atac_npz, allow_pickle=True)
vecs = [bank[k].astype(np.float32) for k in args.replicate_keys]
mat = np.stack(vecs, axis=0)
rng = np.random.RandomState(args.seed)
ids = np.arange(adata.n_obs) % mat.shape[0] if args.mode == 'round_robin' else rng.randint(0, mat.shape[0], size=adata.n_obs)
atac_feat = mat[ids][:, :args.project_dim] if args.project_dim > 0 and mat.shape[1] > args.project_dim else mat[ids]
adata.obsm['atac_feat'] = atac_feat
adata.obs['atac_replicate_id'] = ids.astype(str)
adata.obs['atac_context_type'] = args.species
adata.obs['atac_is_condition_level'] = True
adata.uns['atac_note'] = ('ATAC features are replicate-level promoter gene-activity vectors. No replicate averaging is used. Each scRNA cell is assigned one control ATAC replicate. These are not true per-cell paired ATAC profiles.')
adata.write_h5ad(args.h5ad_out)
