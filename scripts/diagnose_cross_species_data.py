import argparse, json, numpy as np, scanpy as sc
p=argparse.ArgumentParser(); p.add_argument('--human_h5ad',required=True); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--out_json',default='cross_species_qc.json'); a=p.parse_args()
h=sc.read_h5ad(a.human_h5ad); m=sc.read_h5ad(a.mouse_h5ad)
hp=set(h.obs[a.perturb_col].astype(str)); mp=set(m.obs[a.perturb_col].astype(str));
out={'human_cells':int(h.n_obs),'mouse_cells':int(m.n_obs),'human_perturbations':len(hp),'mouse_perturbations':len(mp),'overlap_perturbations':sorted(list(hp&mp)),'missing_mouse_perts':sorted(list(mp-hp))}
if 'atac_feat' in h.obsm: out['human_atac_zero_ratio']=float(np.mean(np.asarray(h.obsm['atac_feat'])==0))
if 'atac_feat' in m.obsm: out['mouse_atac_zero_ratio']=float(np.mean(np.asarray(m.obsm['atac_feat'])==0))
json.dump(out,open(a.out_json,'w'),indent=2)
