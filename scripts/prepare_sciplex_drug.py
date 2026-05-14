import argparse, numpy as np, pandas as pd, scanpy as sc

OOD_TOKENS={'ood','test','holdout'}

def std_split(vals, control_mask, rng):
    arr=np.array([str(x).lower() for x in vals],dtype=object)
    out=np.array(['train']*len(arr),dtype=object)
    out[np.isin(arr,list(OOD_TOKENS)|{'val','valid','validation'})]='test'
    out[np.isin(arr,['train'])]='train'
    out[np.isin(arr,['test'])]='test'
    out[control_mask]='train'
    tr=np.where((out=='train') & (~control_mask))[0]
    if len(tr)>10:
        v=rng.choice(tr,size=max(1,int(0.1*len(tr))),replace=False); out[v]='val'
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--max_cells',type=int,default=None); p.add_argument('--seed',type=int,default=42); p.add_argument('--force_controls_train',action='store_true',default=True)
    a=p.parse_args(); rng=np.random.default_rng(a.seed); ad=sc.read_h5ad(a.input)
    if a.max_cells and ad.n_obs>a.max_cells: ad=ad[rng.choice(ad.n_obs,a.max_cells,replace=False)].copy()
    obs=ad.obs.copy()
    ctrl=((obs.get('control',0).astype(float).values>0)|(obs.get('vehicle',0).astype(float).values>0)|(obs['condition'].astype(str).str.lower().values=='control'))
    obs['is_control']=ctrl.astype(np.int64)
    obs['perturbation']=obs['condition'].astype(str).values; obs.loc[ctrl,'perturbation']='control'
    obs['smiles']=obs['SMILES'].astype(str).values
    dose=pd.to_numeric(obs['dose'],errors='coerce').fillna(0).astype(np.float32).values; dose[ctrl]=0.0; obs['dose']=dose
    ctx=obs['cell_type'].astype(str)
    miss=(ctx.isna()|(ctx=='nan')|(ctx==''))
    if miss.any():
        fallback=obs['cov_drug_dose_name'].astype(str).str.split('_').str[0]
        ctx=ctx.where(~miss,fallback)
    obs['cell_context']=ctx.astype(str); obs['cell_line']=obs['cell_context']
    obs['condition_key']=obs['cov_drug_dose_name'].astype(str); obs['drug_key']=obs['cov_drug'].astype(str)
    obs['perturbation_type']='drug'
    for c in ['target','pathway','pathway_level_1','pathway_level_2','split_random','split_ood','split_ho_pathway','split_tyrosine_ood','split_epigenetic_ood','split_cellcycle_ood']:
        if c not in obs: obs[c]=''
    for src,dst in [('split_random','split_random_std'),('split_ood','split_ood_std'),('split_ho_pathway','split_ho_pathway_std'),('split_epigenetic_ood','split_epigenetic_ood_std'),('split_cellcycle_ood','split_cellcycle_ood_std')]:
        obs[dst]=std_split(obs[src].values, ctrl if a.force_controls_train else np.zeros_like(ctrl), rng)
    ad.obs=obs
    # checks
    print('shape',ad.shape,'perturbations',obs['perturbation'].nunique(),'drugs_no_control',len(set(obs['perturbation'])-{'control'}),'smiles',obs['smiles'].nunique())
    print('dose',pd.Series(obs['dose']).value_counts().head())
    print('context',obs['cell_context'].value_counts().head())
    print('target top10',obs['target'].astype(str).value_counts().head(10))
    for c in [x for x in obs.columns if x.endswith('_std')]: print(c,obs[c].value_counts())
    bad=[]
    for ctxv,df in obs.groupby('cell_context'):
        if ((df['split_ood_std']=='train')&(df['is_control']==1)).sum()==0: bad.append(ctxv)
    if bad: raise ValueError(f'contexts without train control: {bad}')
    ad.write_h5ad(a.output); print('saved',a.output)
if __name__=='__main__': main()
