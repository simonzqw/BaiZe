import torch
import torch.nn as nn
class SourceDeltaResidualPredictor(nn.Module):
    def __init__(self,n_genes,n_perturb_genes,perturb_dim=200,atac_dim=256,use_atac=True,hidden_dim=512,dropout=0.1,residual_scale=0.2):
        super().__init__(); self.use_atac=use_atac; self.residual_scale=residual_scale
        self.rna_encoder=nn.Sequential(nn.Linear(n_genes,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,perturb_dim),nn.LayerNorm(perturb_dim))
        self.perturb_gene_embedding=nn.Embedding(n_perturb_genes,perturb_dim)
        self.atac_encoder=nn.Sequential(nn.Linear(atac_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,perturb_dim),nn.LayerNorm(perturb_dim)) if use_atac else None
        self.fusion=nn.Sequential(nn.Linear(perturb_dim*3,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,n_genes))
        self.gate=nn.Sequential(nn.Linear(perturb_dim*3,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,n_genes),nn.Sigmoid())
    def forward(self,rna_control,source_delta,perturb_gene_idx,target_rna=None,atac_feat=None,return_details=False):
        zr=self.rna_encoder(rna_control); zg=self.perturb_gene_embedding(perturb_gene_idx); za=self.atac_encoder(atac_feat) if (self.use_atac and atac_feat is not None) else torch.zeros_like(zr); z=torch.cat([zr,zg,za],dim=1)
        residual=self.residual_scale*self.gate(z)*self.fusion(z); pred_delta=source_delta+residual; pred_target=rna_control+pred_delta
        if target_rna is None: return pred_target
        true_delta=target_rna-rna_control; residual_target=true_delta-source_delta
        lp=torch.mean((pred_target-target_rna)**2); lr=torch.mean((residual-residual_target)**2); lrn=torch.mean(torch.norm(residual,p=2,dim=1)**2)
        k=min(50,true_delta.shape[1]); idx=torch.topk(true_delta.abs(),k=k,dim=1).indices; lt=torch.mean((torch.gather(pred_delta,1,idx)-torch.gather(true_delta,1,idx))**2)
        loss=lp+lr+0.5*lt+1e-4*lrn
        return (loss,{'pred_target':pred_target,'pred_delta':pred_delta,'source_delta':source_delta,'residual':residual,'true_delta':true_delta}) if return_details else loss
