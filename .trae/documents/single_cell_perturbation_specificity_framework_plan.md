# Single Cell Perturbation Specificity Modeling Framework (scERso) Development Plan

We will build a simple and efficient MLP discriminator framework for learning the "perturbation-cell background" matching relationship.

## 1. Environment preparation and project initialization
- Create basic directory structure: `models/`, `utils/`, `data/`.
- Write `synthetic_data.py`: Generate simulated single-cell expression matrices (HVGs) and perturbation annotation data to ensure that the process can be run through without real data.

## 2. Core module development
### **Data Layer**
- Implement `PerturbationDataset`: Encapsulate (RNA-seq, Perturbation, Label) triplet.
- Processing feature fusion: splicing high-dimensional RNA features with perturbation vectors (Embedding or One-hot).

### **Model Layer**
- Design `SpecificityMLP`:
    - Input layer: receives the fused features.
    - Hidden layer: multi-layer fully connected + BatchNorm + Dropout (to prevent over-fitting).
    - Output layer: Sigmoid activation, output matching probability.

### **Train Layer**
- Write `train.py`: including a complete training cycle, early stopping mechanism (Early Stopping) and performance indicator (AUC, Accuracy) monitoring.

## 3. Verification and Delivery
- Run tests on synthetic data to ensure the model converges and correctly identifies "specific" patterns.
- Provide detailed documentation on how to replace HVGs with real large-scale pre-training Embedding.

## Key technical points
- **Feature Dimensionality Reduction**: Initially use HVGs (such as 2000 dimensions) as RNA input.
- **Positive and negative sample balance**: Use a negative sampling strategy when constructing "mismatching" samples.
- **Scalability**: The model interface is compatible with the Embedding input of pre-trained models such as scGPT.

Please confirm whether to start executing according to this plan?
