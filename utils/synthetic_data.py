import numpy as np
import pandas as pd
import torch

def generate_synthetic_data(n_cells=1000, n_genes=2000, n_perturbations=10, n_cell_types=5):
    """
Generate simulated single-cell perturbation data.
    
logic:
1. Randomly assign a Cell Type to each cell.
2. Generate base expression matrices (HVGs).
3. Generate positive samples (Label=1): Perturbation matches the true biological response of the cell type.
4. Generate negative samples (Label=0): randomly replace perturbations to simulate mismatching situations.
    """
    print(f"Generating synthetic data: {n_cells} cells, {n_genes} genes...")
    
    # 1. Randomly assign cell types
    cell_types = np.random.randint(0, n_cell_types, size=n_cells)
    
    # 2. Generate basic expression matrix (simulating HVGs)
    # Generate different base expression distributions for different Cell Types
    rna_data = np.zeros((n_cells, n_genes), dtype=np.float32)
    for ct in range(n_cell_types):
        mask = (cell_types == ct)
        if np.any(mask):
            # Each cell type has a unique "feature mean vector"
            ct_mean = np.random.randn(n_genes) * 2.0
            rna_data[mask] = ct_mean + np.random.randn(np.sum(mask), n_genes)
    
    # 3. Define perturbation
    # Perturbation ID 0-9
    perturbations = np.random.randint(0, n_perturbations, size=n_cells)
    
    # 4. Generate labels (Label)
    # Simulate a certain pattern: only when (perturbation_id + cell_type) is an even number, the reaction (Label=1) will occur
    # This is a simple non-linear law for MLP to learn
    labels = ((perturbations + cell_types) % 2 == 0).astype(np.float32)
    
    # To increase the difficulty, add a little noise to the labels
    noise = np.random.choice([0, 1], size=n_cells, p=[0.95, 0.05])
    labels = np.abs(labels - noise) # Flip 5% of labels
    
    # 5. Convert to DataFrame or Tensor format
    data = {
        'rna': rna_data,
        'perturb': perturbations,
        'cell_type': cell_types,
        'label': labels
    }
    
    return data

def prepare_tensors(data, test_size=0.2):
    """Divide the data into training and validation sets and convert to PyTorch Tensor"""
    rna = data['rna']
    perturb = data['perturb']
    labels = data['label']
    
    # Partition the dataset (manual implementation)
    n_samples = len(labels)
    idx = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    val_size = int(n_samples * test_size)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    train_data = {
        'rna': torch.tensor(rna[train_idx]),
        'perturb': torch.tensor(perturb[train_idx]),
        'label': torch.tensor(labels[train_idx])
    }
    
    val_data = {
        'rna': torch.tensor(rna[val_idx]),
        'perturb': torch.tensor(perturb[val_idx]),
        'label': torch.tensor(labels[val_idx])
    }
    
    return train_data, val_data

if __name__ == "__main__":
    data = generate_synthetic_data()
    train_data, val_data = prepare_tensors(data)
    print(f"Train samples: {len(train_data['label'])}")
    print(f"Val samples: {len(val_data['label'])}")
    print(f"Label distribution (Train): {np.bincount(train_data['label'].numpy().astype(int))}")
