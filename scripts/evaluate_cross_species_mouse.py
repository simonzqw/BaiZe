import argparse, json, numpy as np, scanpy as sc
from scipy.sparse import issparse

CONTROL_ALIASES={"CTRL","CTRL1","ctrl","Control","vehicle","Vehicle"}
def normalize_perturbation(arr, control_key='control'):
    arr=np.asarray(arr,dtype=object); arr=np.array([str(x) for x in arr],dtype=object)
    arr[np.isin(arr,list(CONTROL_ALIASES|{control_key}))]=str(control_key); return arr

def safe_pearson(x,y): return 0.0 if (np.std(x)<1e-8 or np.std(y)<1e-8) else float(np.corrcoef(x,y)[0,1])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--pred_npz',required=True); p.add_argument('--control_key',default='control'); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--out_json',default='metrics_cross_species.json'); args=p.parse_args()
    ad=sc.read_h5ad(args.mouse_h5ad); X=ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X); pred=np.load(args.pred_npz,allow_pickle=True)
    pert=normalize_perturbation(ad.obs[args.perturb_col].values,args.control_key); cidx=np.where(pert==args.control_key)[0]; cmean=X[cidx].mean(0)
    out={}
    for k in pred.files:
        if '|' in k: continue
        pidx=np.where(pert==k)[0]
        if len(pidx)==0: continue
        td=X[pidx].mean(0)-cmean; pd=pred[k]-cmean
        out[k]={'pearson_delta':safe_pearson(pd,td),'mse_delta':float(np.mean((pd-td)**2))}
    out['mean_pearson_delta']=float(np.mean([v['pearson_delta'] for v in out.values()])) if out else 0.0
    with open(args.out_json,'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
