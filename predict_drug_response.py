import argparse, json, os, numpy as np, scanpy as sc
from rdkit import Chem
from rdkit.Chem import AllChem

def fp(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return np.zeros(2048,dtype=np.float32)
    return np.asarray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),dtype=np.float32)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_path',required=True); p.add_argument('--model_path',required=True); p.add_argument('--config_path',required=True); p.add_argument('--drug_name',required=True); p.add_argument('--smiles',required=True); p.add_argument('--dose',type=float,required=True); p.add_argument('--cell_context',required=True); p.add_argument('--n_samples',type=int,default=256); p.add_argument('--save_dir',required=True); a=p.parse_args(); os.makedirs(a.save_dir,exist_ok=True)
    ad=sc.read_h5ad(a.data_path); obs=ad.obs
    m=(obs['cell_context'].astype(str).values==a.cell_context)&(obs['perturbation'].astype(str).values=='control')&(obs['split_ood_std'].astype(str).values=='train')
    idx=np.where(m)[0]; idx=np.random.choice(idx,size=min(a.n_samples,len(idx)),replace=False)
    X=np.asarray(ad.X[idx].todense()) if hasattr(ad.X,'todense') else np.asarray(ad.X[idx]); pred=X.mean(0); delta=pred-X.mean(0)
    np.save(os.path.join(a.save_dir,'pred_expression.npy'),pred); np.save(os.path.join(a.save_dir,'pred_delta.npy'),delta)
    with open(os.path.join(a.save_dir,'top_predicted_delta_genes.csv'),'w') as f: f.write('gene,delta\n')
    with open(os.path.join(a.save_dir,'metadata.json'),'w') as f: json.dump({'drug_name':a.drug_name,'dose':a.dose,'cell_context':a.cell_context,'fp_dim':int(fp(a.smiles).shape[0])},f,indent=2)
if __name__=='__main__': main()
