import argparse, json, os, numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import issparse
safe=lambda x,y:0.0 if (np.std(x)<1e-8 or np.std(y)<1e-8) else float(np.corrcoef(x,y)[0,1])
p=argparse.ArgumentParser(); p.add_argument('--mouse_h5ad',required=True); p.add_argument('--pred_npz',required=True); p.add_argument('--perturb_col',default='perturbation'); p.add_argument('--context_col',default='cell_context'); p.add_argument('--control_key',default='control'); p.add_argument('--out_dir',required=True); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
ad=sc.read_h5ad(a.mouse_h5ad); X=ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X); pred=np.load(a.pred_npz,allow_pickle=True); pert=ad.obs[a.perturb_col].astype(str).values; ctx=ad.obs[a.context_col].astype(str).values
rows=[]
for k in pred.files:
 if '|' not in k: continue
 g,c=k.split('|',1); tidx=np.where((pert==g)&(ctx==c))[0]; cidx=np.where((pert==a.control_key)&(ctx==c))[0]
 if len(tidx)==0 or len(cidx)==0: continue
 t=X[tidx].mean(axis=0); ctrl=X[cidx].mean(axis=0); pvec=np.asarray(pred[k]); pvec=pvec.mean(axis=0) if pvec.ndim==2 else pvec; td=t-ctrl; pdv=pvec-ctrl; top=np.argsort(np.abs(td))[-min(20,len(td)):]
 rows.append({'key':k,'perturbation':g,'context':c,'top20_delta_pearson':safe(pdv[top],td[top]),'top20_mse':float(np.mean((pdv[top]-td[top])**2))})
df=pd.DataFrame(rows); df.to_csv(os.path.join(a.out_dir,'context_eval_full.csv'),index=False); sm=df.groupby('perturbation',as_index=False).mean(numeric_only=True); sm.to_csv(os.path.join(a.out_dir,'context_eval_summary.csv'),index=False); json.dump({'macro_top20_delta_pearson':float(df['top20_delta_pearson'].mean()) if len(df)>0 else 0.0},open(os.path.join(a.out_dir,'context_eval_summary.json'),'w'),indent=2)
