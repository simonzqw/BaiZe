import torch, torch.nn.functional as F

def safe_corr_loss(pred_delta,true_delta,weight=None,eps=1e-8):
    if weight is not None: pred_delta,pred_true=pred_delta*weight,true_delta*weight; true_delta=pred_true
    px=pred_delta-pred_delta.mean(dim=1,keepdim=True); tx=true_delta-true_delta.mean(dim=1,keepdim=True)
    return 1.0-((px*tx).sum(dim=1)/(torch.sqrt((px**2).sum(dim=1)+eps)*torch.sqrt((tx**2).sum(dim=1)+eps))).mean()

def weighted_mse(pred,target,weight=None): return F.mse_loss(pred,target) if weight is None else ((pred-target)**2*weight).mean()
def sign_consistency_loss(pred_delta,true_delta,weight=None):
    p=F.relu(-(pred_delta*true_delta)); return (p*weight).mean() if weight is not None else p.mean()
def delta_norm_loss(pred_delta,true_delta,eps=1e-8):
    return F.mse_loss(torch.sqrt((pred_delta**2).sum(dim=1)+eps),torch.sqrt((true_delta**2).sum(dim=1)+eps))
def topk_corr_loss(pred_delta,true_delta,k=20):
    k=min(k,true_delta.shape[1]); idx=torch.topk(true_delta.abs(),k=k,dim=1).indices
    return safe_corr_loss(torch.gather(pred_delta,1,idx),torch.gather(true_delta,1,idx))
def cross_species_metric_loss(pred,target,pred_delta,true_delta,residual,gene_weight=None,mse_weight=1.0,delta_mse_weight=1.0,corr_weight=0.5,topk_corr_weight=0.5,sign_weight=0.2,norm_weight=0.1,residual_l2_weight=0.01,topk=20):
    lp=weighted_mse(pred,target,gene_weight); ld=weighted_mse(pred_delta,true_delta,gene_weight); lc=safe_corr_loss(pred_delta,true_delta,gene_weight); ltc=topk_corr_loss(pred_delta,true_delta,topk); ls=sign_consistency_loss(pred_delta,true_delta,gene_weight); ln=delta_norm_loss(pred_delta,true_delta); lr=(residual**2).mean(); total=mse_weight*lp+delta_mse_weight*ld+corr_weight*lc+topk_corr_weight*ltc+sign_weight*ls+norm_weight*ln+residual_l2_weight*lr
    return total,{"loss":float(total.detach().cpu()),"loss_pred":float(lp.detach().cpu()),"loss_delta":float(ld.detach().cpu()),"loss_corr":float(lc.detach().cpu()),"loss_topk_corr":float(ltc.detach().cpu()),"loss_sign":float(ls.detach().cpu()),"loss_norm":float(ln.detach().cpu()),"loss_residual":float(lr.detach().cpu())}
