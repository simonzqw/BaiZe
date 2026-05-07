import argparse, json, os, numpy as np, scanpy as sc
from scipy.sparse import issparse
p=argparse.ArgumentParser(); p.add_argument('--combined_h5ad',required=True); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--perturbations',nargs='+',required=True); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--species_col',default='species'); p.add_argument('--human_value',default='human'); p.add_argument('--control_key',default='control'); p.add_argument('--context_specific',action='store_true'); p.add_argument('--context_col',default='cell_context'); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
c=sc.read_h5ad(a.combined_h5ad); m=sc.read_h5ad(a.mouse_h5ad); Xc=c.X.toarray() if issparse(c.X) else np.asarray(c.X); Xm=m.X.toarray() if issparse(m.X) else np.asarray(m.X)
cp=c.obs[a.perturb_col].astype(str).values; cp[np.isin(cp,['CTRL','CTRL1','ctrl','Control','vehicle','Vehicle'])]=a.control_key; cs=c.obs[a.species_col].astype(str).values
hmask=cs==a.human_value; hctrl=Xc[hmask&(cp==a.control_key)].mean(axis=0); preds={}; bank={}
for g in a.perturbations:
 idx=np.where(hmask&(cp==g))[0]
 if len(idx)==0: continue
 d=Xc[idx].mean(axis=0)-hctrl; bank[g]=d.astype(np.float32); mp=m.obs[a.perturb_col].astype(str).values; mp[np.isin(mp,['CTRL','CTRL1','ctrl','Control','vehicle','Vehicle'])]=a.control_key; cidx=np.where(mp==a.control_key)[0]
 if len(cidx)>0: preds[g]=(Xm[cidx].mean(axis=0)+d).astype(np.float32)
np.savez_compressed(os.path.join(a.out_dir,'mouse_cross_species_preds.npz'),**preds); np.savez_compressed(os.path.join(a.out_dir,'source_delta_bank.npz'),**bank); json.dump({'n_preds':len(preds)},open(os.path.join(a.out_dir,'meta.json'),'w'),indent=2)
