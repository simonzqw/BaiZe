import torch
import torch.nn as nn

class SpecificityMLP(nn.Module):
    """
Single-cell perturbation-specific discriminant model (MLP) - enhanced version
    
support:
1. RNA-seq signatures (HVGs)
2. Perturbation of Gene Embedding
3. Cell line/tissue context Embedding (to achieve specific modeling required by the teacher)
    """
    def __init__(self, n_genes, n_perturbations, n_cell_lines, 
                 perturb_dim=128, cell_line_dim=32, 
                 hidden_dims=[512, 256, 128], dropout=0.2):
        super(SpecificityMLP, self).__init__()
        
        # 1. Embedding layer
        self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)
        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)
        
        # 2. Feature fusion layer
        # Input dimension = RNA dimension + perturbation dimension + cell line dimension
        input_dim = n_genes + perturb_dim + cell_line_dim
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        # 3. Output layer (two categories: matching/responsive vs. mismatch/normal)
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, rna, perturb, cell_line):
        """
        Args:
            rna: [batch_size, n_genes]
perturb: [batch_size] perturbation ID
cell_line: [batch_size] cell line ID
        """
        p_emb = self.perturb_embedding(perturb)
        c_emb = self.cell_line_embedding(cell_line)
        
        # Splice all features
        x = torch.cat([rna, p_emb, c_emb], dim=1)
        
        return self.network(x)
