from torch.utils.data import Dataset, DataLoader

class PerturbationDataset(Dataset):
    """
Single cell perturbation data set
    """
    def __init__(self, rna_tensors, perturb_tensors, label_tensors):
        """
        Args:
rna_tensors (torch.Tensor): expression matrix
perturb_tensors (torch.Tensor): perturbation ID
label_tensors (torch.Tensor): labels (1=match, 0=no match)
        """
        self.rna = rna_tensors
        self.perturb = perturb_tensors
        self.label = label_tensors
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return {
            'rna': self.rna[idx],
            'perturb': self.perturb[idx].long(), # Make sure it is a long integer for Embedding
            'label': self.label[idx].float()
        }

def get_dataloader(data_dict, batch_size=32, shuffle=True):
    dataset = PerturbationDataset(
        data_dict['rna'], 
        data_dict['perturb'], 
        data_dict['label']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
