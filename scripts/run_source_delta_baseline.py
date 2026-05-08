import argparse, os, numpy as np, scanpy as sc
from scipy.sparse import issparse
CONTROL_ALIASES={"CTRL","CTRL1","ctrl","Control","vehicle","Vehicle"}
def normalize_perturbation(arr, control_key='control'):
    arr=np.asarray(arr,dtype=object); arr=np.array([str(x) for x in arr],dtype=object)
    arr[np.isin(arr,list(CONTROL_ALIASES|{control_key}))]=str(control_key); return arr

def main():
    p=argparse.ArgumentParser(); p.add_argument('--combined_h5ad',required=True); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--perturbations',nargs='+',required=True); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--species_col',default='species'); p.add_argument('--human_value',default='human'); p.add_argument('--control_key',default='control'); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    c=sc.read_h5ad(a.combined_h5ad); m=sc.read_h5ad(a.mouse_h5ad); Xc=c.X.toarray() if issparse(c.X) else np.asarray(c.X)
    cp=normalize_perturbation(c.obs[a.perturb_col].values,a.control_key); cs=c.obs[a.species_col].astype(str).values; hm=(cs==a.human_value); hctrl=Xc[hm & (cp==a.control_key)].mean(0)
    bank={}
    for g in a.perturbations:
        idx=np.where(hm & (cp==g))[0]
        if len(idx)==0: continue
        bank[g]=(Xc[idx].mean(0)-hctrl).astype(np.float32)
    np.savez_compressed(os.path.join(a.out_dir,'mouse_cross_species_preds.npz'),**bank)
if __name__=='__main__': main()
