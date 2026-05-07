import argparse, json, numpy as np, scanpy as sc

def safe_pearson(x,y):
    return 0.0 if (np.std(x)<1e-8 or np.std(y)<1e-8) else float(np.corrcoef(x,y)[0,1])

p=argparse.ArgumentParser(); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--pred_npz',required=True); p.add_argument('--control_key',default='control'); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--context_col',default='cell_context'); p.add_argument('--output_json',default='cross_species_mouse_eval.json'); a=p.parse_args()
adata=sc.read_h5ad(a.mouse_h5ad); pred=np.load(a.pred_npz,allow_pickle=True); y=np.asarray(adata.X); pert=adata.obs[a.perturb_col].astype(str).values; ctx=adata.obs[a.context_col].astype(str).values if a.context_col in adata.obs else np.array(['global']*adata.n_obs)
pert[np.isin(pert,['CTRL','CTRL1','ctrl','Control','vehicle','Vehicle'])]=a.control_key
out={}
for k in pred.files:
    idx=np.where(pert==k)[0]
    if len(idx)==0: continue
    cands=np.where(pert==a.control_key)[0]
    if len(cands)==0: continue
    ctrl=y[cands].mean(axis=0); t=y[idx].mean(axis=0); pvec=np.asarray(pred[k]).mean(axis=0); td=t-ctrl; pd=pvec-ctrl; top=np.argsort(np.abs(td))[-min(20,len(td)):]
    out[k]={'pearson_all':safe_pearson(pvec,t),'pearson_delta':safe_pearson(pd,td),'top20_delta_pearson':safe_pearson(pd[top],td[top]),'top20_delta_mse':float(np.mean((pd[top]-td[top])**2)),'opposite_direction_top20':float(np.mean(np.sign(pd[top])!=np.sign(td[top])))}
json.dump(out,open(a.output_json,'w'),indent=2)
