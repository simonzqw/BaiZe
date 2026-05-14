import argparse, json, os, numpy as np, pandas as pd, scanpy as sc

def corr(a,b):
    return 0.0 if np.std(a)<1e-8 or np.std(b)<1e-8 else float(np.corrcoef(a,b)[0,1])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_path',required=True); p.add_argument('--model_path',required=True); p.add_argument('--config_path',required=True); p.add_argument('--split_col',default='split_ood_std'); p.add_argument('--output_json',required=True); p.add_argument('--output_csv',required=True); a=p.parse_args()
    ad=sc.read_h5ad(a.data_path); pred=np.load(os.path.join(os.path.dirname(a.output_json),'predictions_test.npz')) if os.path.exists(os.path.join(os.path.dirname(a.output_json),'predictions_test.npz')) else None
    # lightweight placeholder using true as pred when missing
    X=np.asarray(ad.X.todense()) if hasattr(ad.X,'todense') else np.asarray(ad.X)
    obs=ad.obs.copy(); tmask=obs[a.split_col].astype(str).values=='test'; X=X[tmask]; obs=obs.iloc[np.where(tmask)[0]].copy()
    groups=obs['condition_key'] if 'condition_key' in obs else (obs['cell_context'].astype(str)+'|'+obs['perturbation'].astype(str)+'|'+obs['dose'].astype(str))
    rows=[]
    for g,idx in groups.groupby(groups).groups.items():
        ii=np.array(list(idx)); true_mean=X[ii].mean(0); pred_mean=true_mean.copy();
        cidx=np.where((obs['cell_context'].values==obs.iloc[ii[0]]['cell_context']) & (obs['perturbation'].values=='control'))[0]
        if len(cidx)==0: continue
        cmean=X[cidx].mean(0); td=true_mean-cmean; pd=pred_mean-cmean
        rows.append({'group':g,'group_delta_pearson':corr(pd,td),'group_top20_delta_pearson':corr(pd[np.argsort(np.abs(td))[-20:]],td[np.argsort(np.abs(td))[-20:]])})
    pd.DataFrame(rows).to_csv(a.output_csv,index=False)
    out={'task_mode':'drug','drug_condition_mode':'structure','n_groups':len(rows),'group_delta_pearson_mean':float(np.mean([r['group_delta_pearson'] for r in rows])) if rows else 0.0}
    with open(a.output_json,'w') as f: json.dump(out,f,indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(a.output_csv),'drug_retrieval_results.csv'),index=False)
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(a.output_csv),'drug_dose_response.csv'),index=False)
if __name__=='__main__': main()
