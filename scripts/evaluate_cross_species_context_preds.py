import argparse, json, os, numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import issparse
CONTROL_ALIASES={"CTRL","CTRL1","ctrl","Control","vehicle","Vehicle"}
def normalize_perturbation(arr, control_key='control'):
    arr=np.asarray(arr,dtype=object); arr=np.array([str(x) for x in arr],dtype=object)
    arr[np.isin(arr,list(CONTROL_ALIASES|{control_key}))]=str(control_key); return arr
safe=lambda x,y:0.0 if (np.std(x)<1e-8 or np.std(y)<1e-8) else float(np.corrcoef(x,y)[0,1])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--pred_npz',required=True); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--context_col',default='cell_context'); p.add_argument('--control_key',default='control'); p.add_argument('--out_dir',required=True); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    ad=sc.read_h5ad(a.mouse_h5ad); X=ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X); pred=np.load(a.pred_npz,allow_pickle=True); pert=normalize_perturbation(ad.obs[a.perturb_col].values,a.control_key); ctx=ad.obs[a.context_col].astype(str).values
    rows=[]
    for key in pred.files:
        if '|' not in key: continue
        g,c=key.split('|',1); m=(ctx==c); ppert=pert[m]; XX=X[m]
        cidx=np.where(ppert==a.control_key)[0]; pidx=np.where(ppert==g)[0]
        if len(cidx)==0 or len(pidx)==0: continue
        td=XX[pidx].mean(0)-XX[cidx].mean(0); pd=pred[key]-XX[cidx].mean(0)
        rows.append({'key':key,'pearson_delta':safe(pd,td),'mse_delta':float(np.mean((pd-td)**2))})
    df=pd.DataFrame(rows); df.to_csv(os.path.join(a.out_dir,'context_metrics.csv'),index=False)
    with open(os.path.join(a.out_dir,'context_metrics.json'),'w') as f: json.dump(rows,f,indent=2)
if __name__=='__main__': main()
