import pandas as pd
import numpy as np
import torch
import os

class GeneEmbeddingLoader:
    """
Responsible for loading external pre-trained gene vectors and aligning them with the perturbation ID of the current data
    """
    def __init__(self, embedding_path, perturb_id_to_name_map):
        self.embedding_path = embedding_path
        self.perturb_map = perturb_id_to_name_map
        
    def load_weights(self, default_dim=200):
        if not os.path.exists(self.embedding_path):
            print(f"!!! Warning: Pretrained file {self.embedding_path} does not exist. Random initialization will be used.")
            return None
        
        print(f">>> Loading pretrained vectors: {self.embedding_path}")
        
        # Assume the file is CSV (gene_name, dim1, dim2...), no header or index is the gene name
        # Fine-tuning may be required depending on the actual download format (for example, gene2vec is usually txt or bin)
        try:
            # Gene2Vec .txt format is usually: the first line is (n_genes, dim), or directly gene val1 val2...
            # We use pd.read_csv directly and automatically judge based on the content
            df = pd.read_csv(self.embedding_path, sep='\s+', index_col=0, header=None, engine='python')
            
            # If the first line is metadata (e.g. 24447 200), we need to filter out
            if isinstance(df.index[0], (int, float)) or (isinstance(df.index[0], str) and df.index[0].isdigit()):
                print(">>> It was detected that the first line of the file contains metadata and has been skipped.")
                df = df.iloc[1:]
                
            print(f">>> Loaded {len(df)}  pretrained gene vectors; dimension: {df.shape[1]}")
            
            # Build weight matrix
            n_perturbations = len(self.perturb_map)
            emb_dim = df.shape[1]
            weights = np.random.normal(scale=0.02, size=(n_perturbations, emb_dim))
            mean_vec = df.values.mean(axis=0)
            
            hit_count = 0
            # Loop through all perturbation IDs in our data
            for idx, gene_name in self.perturb_map.items():
                if gene_name == 'control':
                    # Control can be initialized to all zeros or to a specific vector
                    weights[idx] = np.zeros(emb_dim)
                    continue
                    
                # Try to match (handle case inconsistencies, etc.)
                if gene_name in df.index:
                    weights[idx] = df.loc[gene_name].values
                    hit_count += 1
                elif gene_name.upper() in df.index:
                    weights[idx] = df.loc[gene_name.upper()].values
                    hit_count += 1
                else:
                    weights[idx] = mean_vec
                    
            print(f">>> Pretrained-vector matching rate: {hit_count}/{n_perturbations} ({hit_count/n_perturbations:.1%})")
            return torch.FloatTensor(weights)
            
        except Exception as e:
            print(f"!!! Failed to load pretrained vectors: {e}")
            return None
