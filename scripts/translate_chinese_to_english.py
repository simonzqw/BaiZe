from __future__ import annotations

import pathlib
import re
import subprocess

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

README = r'''# scERso Conditional Diffusion Model for Single-cell Response Prediction

The repository's **main workflow now uses the conditional diffusion model**. The legacy MLP/Transformer training path (`train.py`) is retained for historical comparison but is no longer recommended.

Use the following entry points for current experiments:
- Training: `train_diffusion.py`
- Evaluation: `evaluate_diffusion.py`
- Inference, perturbation composition, and interpolation: `predict_diffusion.py`
- Visualization: `visualize_diffusion.py`

---

## 1. Main capabilities

### 1.1 Conditional diffusion backbone
- Background-effect disentanglement: `z_bg` (background) and `z_eff` (effect) are encoded separately.
- Target modes: `target_mode = target | delta`.
- Sampling and guidance: classifier-free guidance, configurable DDIM sampling steps, and EMA-weight evaluation are supported.

### 1.2 Multiple task modes
Use `--task_mode` to select the task definition and avoid mismatches between the task and the input data:

1. `single_gene`
   - Designed for single-gene perturbation datasets such as Adamson.
   - Condition fields: `perturb_gene_idx` and `is_control`.

2. `translation`
   - Designed for two-condition translation tasks, such as day4 -> day6.
   - Condition fields: `condition_id` and `source_flag`.

3. `drug`
   - Designed for drug-response prediction with optional molecular structure, dose, and cellular-context conditions.

### 1.3 Data and control/reference handling
- Supports `split_strategy = random | perturbation | custom`.
- Under perturbation zero-shot splits, validation and test samples reuse the training control bank when their own splits contain no controls.
- Supports `control_match_mode`, `control_prototype_mode`, and `control_prototype_temp`.

---

## 2. Project structure

- `train_diffusion.py`: primary conditional-diffusion training entry point.
- `evaluate_diffusion.py`: single-cell and perturbation-level evaluation.
- `predict_diffusion.py`: single and combinatorial perturbation prediction and latent interpolation.
- `visualize_diffusion.py`: combinatorial perturbation analysis and diagnostic visualization.
- `models/scerso_diffusion.py`: conditional diffusion model definition.
- `models/diffusion_core.py`: diffusion process implementation.
- `utils/data_processor.py`: h5ad loading, split construction, control pools, and condition fields.
- `docs/diffusion_methodology.md`: methodological description.

> Legacy files such as `train.py`, `evaluate_metrics.py`, and `visualize.py` are retained only for historical comparison and are not part of the recommended workflow.

---

## 3. Environment and dependencies

Recommended environment:
- Python 3.8+
- PyTorch
- scanpy / anndata
- numpy / scipy / pandas
- scikit-learn
- matplotlib / seaborn
- rdkit, only when SMILES-derived drug features are used

The following setting is also recommended:

```bash
export OMP_NUM_THREADS=1
```

---

## 4. Training

### 4.1 Adamson (`single_gene`)

```bash
python train_diffusion.py \
  --data_path /path/to/adamson/perturb_processed.h5ad \
  --save_dir ./checkpoints_adamson_single_gene \
  --task_mode single_gene \
  --split_strategy perturbation \
  --preset vnext \
  --amp
```

When the training data contain combinatorial labels such as `double_...`, `triple_...`, or `GENE1+GENE2+GENE3`, enable multi-gene label parsing:

```bash
python train_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --save_dir ./checkpoints_combo_diffusion \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --preset vnext \
  --amp
```

### 4.2 day4/day6 (`translation`)

```bash
python train_diffusion.py \
  --data_path /path/to/day4_to_day6_diffusion.h5ad \
  --save_dir ./checkpoints_day4_day6_translation \
  --task_mode translation \
  --split_strategy custom \
  --split_col split \
  --atac_key atac_feat \
  --preset vnext \
  --amp
```

> Use `--preset smoke` for a quick smoke test.

---

## 5. Evaluation

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --output_json ./checkpoints_xxx/eval_metrics.json
```

A three-gene composition case can also be evaluated. When the corresponding combination label is present in the h5ad file, the output includes metrics against its observed mean response:

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --perturb_parse_mode multi_gene_parse \
  --cell_line K562 \
  --combo_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --output_json ./checkpoints_xxx/eval_metrics_triple.json
```

For translation data, use:

```bash
--task_mode translation --split_strategy custom --split_col split
```

---

## 6. Inference and visualization

### 6.1 Prediction, composition, and interpolation

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --latent_mode adaptive \
  --save_dir ./pred_out
```

For a three-gene perturbation, append the third gene to the original two-gene command:

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_dir ./pred_out_triple
```

### 6.2 Visualization

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --save_path ./combo_report.png
```

Three-gene composition visualization:

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 SOX17 \
  --latent_mode adaptive \
  --save_path ./triple_combo_report.png
```

---

## 7. Frequently asked questions

### Q1: `adata.obs is missing the custom split column: split`
The command uses `split_strategy=custom`, but the data do not contain `obs['split']`. Either use:

```bash
--split_strategy perturbation
```

or create the `split` column in the h5ad file before training.

### Q2: I passed `--split_strategy perturbation`, but the log still reports `custom`
`--preset` only overrides parameters that were not explicitly supplied. Explicit command-line arguments are retained. If the issue persists, check whether the same argument appears more than once in the command.

### Q3: The validation or test split reports an empty control pool
Under a perturbation zero-shot split, validation and test data reuse the training control bank. If the error persists, the training split itself probably contains no control samples and the source data should be checked.

---

## 8. Cross-species perturbation prediction module

This module is independent from the diffusion mainline.

### 8.1 Legacy scripts

- `scripts/prepare_mouse_context.py`
- `scripts/train_cross_species_ctx.py`
- `scripts/cross_species_infer_ctx.py`

### 8.2 Recommended v2 workflow

1. Diagnose data: `scripts/cross_species_diagnose_data.py`
2. Build bootstrap pseudo-bulk: `scripts/cross_species_build_pseudobulk.py`
3. Train residual model: `scripts/cross_species_train_residual.py`
4. Run context-wise inference: `scripts/cross_species_infer_residual.py`
5. Evaluate: `scripts/evaluate_cross_species_mouse.py`, `scripts/evaluate_cross_species_context_preds.py`
'''

METHODOLOGY = r'''# Current scERso Diffusion Methodology

## 1. Task definition and conditional modeling

The model learns the conditional distribution:

\[
p(\mathbf{x}_{\text{pert}}\mid \mathbf{x}_{\text{ctrl}},\ \text{perturb},\ \text{cell\_line},\ \text{dose},\ \text{ATAC},\ \text{drug})
\]

Here, \(\mathbf{x}_{\text{ctrl}}\) is the control RNA-expression profile and \(\mathbf{x}_{\text{pert}}\) is the post-perturbation expression profile.

## 2. Semantic latent variables and context

The model first encodes multimodal conditions into a semantic latent variable \(\mathbf{z}_{sem}\):

- projection of the control RNA profile;
- perturbation embedding, optionally scaled by dose;
- cell-line embedding;
- dose projection;
- optional ATAC and drug-feature projections;
- multi-head self-attention followed by a residual MLP and LayerNorm to produce \(\mathbf{z}_{sem}\);
- a parallel joint semantic encoder that concatenates RNA, perturbation, cell-line, and dose features, processes them with an MLP, and combines them with the attention pathway through a gate to stabilize single-perturbation representations.

The diffusion condition vector is then constructed as:

\[
\mathbf{c}=\left[\mathbf{x}_{\text{ctrl}};\mathbf{z}_{sem}\right]
\]

Conditional dropout is supported during training by randomly setting \(\mathbf{z}_{sem}\) to zero, enabling classifier-free guidance.

## 3. Forward diffusion process

A Gaussian forward diffusion process is used:

\[
q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\right)
\]

It is implemented as:

\[
\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})
\]

The noise schedule can be linear or cosine, with cosine used by default.

## 4. Reverse denoising network

The denoiser is a Squidiff-style MLP:

- it receives \(\mathbf{x}_t\) and the condition \(\mathbf{c}\);
- timestep \(t\) is represented using sinusoidal positional encoding followed by an MLP;
- the time embedding and \(\mathbf{z}_{sem}\) are injected into every residual block;
- the output has the same dimensionality as the expression input.

The current objective is `pred_x0`, meaning that the network directly predicts \(\hat{\mathbf{x}}_0\).

## 5. Training objective

At every iteration, a timestep \(t\) is sampled and the following objective is minimized:

\[
\mathcal{L}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\lVert f_\theta(\mathbf{x}_t,t,\mathbf{c})-\mathbf{x}_0\rVert_2^2\right]
\]

The implementation first averages the loss over features for each sample and then averages over the batch. Optional sample weights can be supplied by the timestep resampler.

## 6. Sampling and inference

### DDPM sampling

The model iterates from \(t=T-1\) to \(0\):

1. predict \(\hat{\mathbf{x}}_0\), or predict noise and convert it to \(\hat{\mathbf{x}}_0\);
2. sample \(\mathbf{x}_{t-1}\) from the posterior mean and variance of \(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\hat{\mathbf{x}}_0)\).

### Fast DDIM sampling

When `sample_steps < timesteps`, a DDIM subsequence is used. The parameter \(\eta\) controls stochasticity.

### Latent interpolation

A linear interpolation trajectory can be constructed between two semantic latent vectors:

\[
z(\alpha)=(1-\alpha)z_A+\alpha z_B,\quad \alpha\in[0,1]
\]

This can be used to analyze continuous dose or state transitions with `predict_diffusion.py --interpolate_to --interp_steps`.

### Classifier-free guidance

Conditional and unconditional predictions are combined as:

\[
\hat{y}=\hat{y}_{uncond}+s(\hat{y}_{cond}-\hat{y}_{uncond})
\]

where \(s\) is `guidance_scale`.

## 7. Mathematical interpretation

1. **High-dimensional expression generation is converted into progressive refinement.** The reverse process starts from isotropic Gaussian noise and gradually contracts toward an expression vector consistent with the conditional distribution.
2. **The conditional latent variable \(\mathbf{z}_{sem}\) acts as a perturbation-semantic coordinate.** Perturbation, cell-line, dose, ATAC, and drug information are represented in one latent space and shape the reverse diffusion trajectory.
3. **The `pred_x0` objective directly supervises the biological signal.** Compared with pure noise prediction, direct supervision of \(\mathbf{x}_0\) provides a more direct fit to expression amplitudes, although it relies on appropriate normalization and calibration.
4. **Classifier-free guidance amplifies the conditional contribution.** It improves condition consistency during sampling, usually at the cost of some diversity.
5. **Timestep resampling follows an importance-sampling principle.** Loss-second-moment sampling focuses on high-loss timesteps, approximately reducing gradient variance and improving sample efficiency.

## 8. Relationship to combinatorial perturbations

The implementation can encode individual perturbation latent vectors, combine them, and then sample the resulting response:

- `sum/mean`: linear composition that is stable and directly interpretable;
- `adaptive`: weighted linear composition augmented with pairwise nonlinear interactions,
  \(\phi([z_i,z_j,z_i\odot z_j,|z_i-z_j|])\), followed by gated fusion:

  \[
  z_{combo}=g\odot z_{lin} + (1-g)\odot (z_{lin}+z_{pair})
  \]

  where \(g=\sigma(\psi([z_{lin},\bar z]))\).

This formulation captures part of the synergistic or antagonistic nonlinear response while preserving the original single-perturbation pathway.
'''

PLAN = r'''# Development Plan for the Single-cell Perturbation Specificity Modeling Framework (scERso)

We will build a simple and efficient MLP discriminator that learns whether a perturbation is compatible with a given cellular background.

## 1. Environment and project initialization

- Create the base directory structure: `models/`, `utils/`, and `data/`.
- Implement `synthetic_data.py` to generate simulated highly variable gene expression matrices and perturbation annotations, allowing the complete workflow to run without real data.

## 2. Core modules

### Data layer

- Implement `PerturbationDataset` to package `(RNA-seq, perturbation, label)` tuples.
- Concatenate high-dimensional RNA features with perturbation vectors represented by embeddings or one-hot encodings.

### Model layer

- Implement `SpecificityMLP`:
  - Input layer: receives the fused feature vector.
  - Hidden layers: fully connected layers with BatchNorm and Dropout to reduce overfitting.
  - Output layer: applies a sigmoid activation and returns a matching probability.

### Training and validation layer

- Implement `train.py` with a complete training loop, early stopping, and monitoring of AUC and accuracy.

## 3. Validation and delivery

- Run tests on synthetic data to verify convergence and correct identification of specificity patterns.
- Provide detailed documentation explaining how to replace highly variable genes with large pretrained embeddings.

## Key technical considerations

- **Feature dimensionality reduction:** initially use highly variable genes, such as a 2,000-dimensional RNA input.
- **Positive-negative balance:** use negative sampling when constructing mismatched examples.
- **Extensibility:** keep the model interface compatible with embeddings produced by pretrained models such as scGPT.
'''

REPLACEMENTS = {
    "没有可用的 cell-line control baseline。": "No cell-line control baseline is available.",
    "未传 --cell_line，默认使用": "--cell_line was not provided; using the default",
    "未找到 cell line": "Cell line not found",
    "--combo_genes 至少需要两个基因；三基因 case 例如": "--combo_genes requires at least two genes; for a three-gene case, use",
    "组合扰动基因不在 perturbation vocab 中": "The combinatorial perturbation gene is not in the perturbation vocabulary",
    "尽量从 checkpoint 恢复语义 perturb embedding 模式": "Restore the semantic perturbation-embedding mode from the checkpoint when possible",
    "提示: checkpoint 缺少以下参数（已用随机初始化兼容）": "Note: the checkpoint is missing the following parameters; random initialization was used for compatibility",
    "提示: checkpoint 存在未使用参数": "Note: the checkpoint contains unused parameters",
    "指标已保存到": "Metrics saved to",
    "兼容旧 checkpoint：存在 perturb_encoder 参数但没有 perturb_feature_bank": "Legacy checkpoint compatibility: perturb_encoder parameters are present but perturb_feature_bank is absent",
    "这时用 perturb_embedding.weight 作为 feature_bank 构建语义分支结构，确保能完整加载权重。": "Use perturb_embedding.weight as the feature bank to construct the semantic branch so that all checkpoint weights can be loaded.",
    "单细胞扰动特异性判别模型 (MLP) - 增强版": "Single-cell perturbation specificity classifier (MLP), enhanced version",
    "支持:": "Supports:",
    "RNA-seq 特征 (HVGs)": "RNA-seq features (HVGs)",
    "扰动基因 Embedding": "Perturbation-gene embeddings",
    "细胞系/组织上下文 Embedding (实现老师要求的特异性建模)": "Cell-line or tissue-context embeddings for context-specific modeling",
    "嵌入层": "Embedding layers",
    "特征融合层": "Feature-fusion layers",
    "输入维度 = RNA维度 + 扰动维度 + 细胞系维度": "Input dimension = RNA dimension + perturbation dimension + cell-line dimension",
    "输出层 (二分类: 匹配/响应 vs 不匹配/正常)": "Output layer (binary classification: matched/response versus mismatched/baseline)",
    "扰动 ID": "perturbation ID",
    "细胞系 ID": "cell-line ID",
    "拼接所有特征": "Concatenate all features",
    "冻结": "frozen",
    "解冻": "unfrozen",
    "扰动 Embedding 层已": "Perturbation embedding layer is",
    "target_mode 必须是 'target' 或 'delta'": "target_mode must be either 'target' or 'delta'",
    "latents 不能为空。": "latents cannot be empty.",
    "weights 长度必须和 latents 一致。": "The length of weights must match the number of latents.",
    "未知组合模式": "Unknown composition mode",
    "steps 必须 >= 2。": "steps must be >= 2.",
    "未找到可用的非 control 扰动基因。": "No usable non-control perturbation gene was found.",
    "组合扰动当前默认复用同一 cell-line baseline ATAC；若组合引发显著染色质变化，建议外部构建组合特异 ATAC 条件。": "Combinatorial perturbations currently reuse the same cell-line baseline ATAC profile. If a combination causes substantial chromatin changes, construct a combination-specific ATAC condition externally.",
    "扰动": "Perturbation",
    "不存在，自动替换为": "was not found and was automatically replaced with",
    "已匹配真实均值样本": "Matched observed mean samples",
    "插值轨迹保存": "Interpolation trajectory saved to",
    "预测 CSV": "Prediction CSV",
    "可视化图": "Visualization figure",
    "latent 保存": "Latent representation saved to",
    "GlobalMSE 权重": "GlobalMSE weight",
    "DeltaMSE 放大系数": "DeltaMSE scaling factor",
    "DeltaMSE (带加权)": "Weighted DeltaMSE",
    "根据真实变化量的绝对值进行加权，强迫模型拟合剧烈变化的基因": "Weight genes by the absolute observed change to emphasize strongly responding genes",
    "归一化权重": "Normalize weights",
    "GlobalMSE (基础保底)": "GlobalMSE as a baseline safeguard",
    "Control 正则化 (如果输入是 control 样本，则 delta 必须为 0)": "Control regularization: delta must be zero for control inputs",
    "优先只在 control 样本上施加约束，避免过度抑制真实扰动信号": "Apply this constraint only to control samples to avoid suppressing genuine perturbation signals",
    "注意：现在监控的是 Top50 Pearson，越高越好": "Monitor Top50 Pearson; higher values are better",
    "模型架构参数": "Model architecture parameters",
    "返回更完整的指标": "Return a more complete set of metrics",
    "全谱": "Genome-wide",
    "对所有非dropout(真实delta!=0)基因统计": "Compute statistics over all non-dropout genes with observed delta != 0",
    "聚合": "Aggregate",
    "scERso V7 启动 | 任务: 生成式扰动预测 | 策略": "scERso V7 started | task: generative perturbation prediction | strategy",
    "数据准备": "Data preparation",
    "当前是 unseen perturbation 划分，必须提供 side information（如 --pretrained_emb 或 SMILES 药物特征），": "The current split contains unseen perturbations, so side information such as --pretrained_emb or SMILES-derived drug features is required,",
    "否则 perturbation 只能依赖 ID embedding，泛化会明显受限。": "otherwise perturbations rely only on ID embeddings and generalization will be substantially limited.",
    "加载预训练向量": "Load pretrained vectors",
    "初始化模型": "Initialize the model",
    "新增药物维度参数": "Added drug-feature dimension",
    "损失函数: 加权 MSE (根据 Delta 绝对值加权，抑制基线红利)": "Loss: weighted MSE using absolute delta to reduce baseline-dominated performance",
    "参数分组优化": "Parameter-group optimization",
    "训练循环": "Training loop",
    "监控 -Top20 MSE，越大越好（等价于 Top20 MSE 越小越好）": "Monitor negative Top20 MSE; larger is better and is equivalent to minimizing Top20 MSE",
    "检测到断点续训": "Resuming from checkpoint",
    "从 epoch": "Continue training from epoch",
    "继续训练": "",
    "冻结策略": "Freezing strategy",
    "层已 冻结": "layer frozen",
    "层已 解冻": "layer unfrozen",
    "药物特征处理 (如果有)": "Process drug features when available",
    "根据 perturb index 获取对应的 drug feature": "Retrieve the corresponding drug feature using the perturbation index",
    "传递 drug_feat 和 dose 到 forward": "Pass drug_feat and dose to the forward method",
    "使用加权损失": "Use the weighted loss",
    "验证集评估": "Validation evaluation",
    "药物特征处理 (验证集)": "Process drug features for validation",
    "计算多维指标": "Compute multidimensional metrics",
    "聚合所有 batch 的指标": "Aggregate metrics across all batches",
    "核心：以 Top20 MSE 作为保存和早停依据（越小越好）": "Use Top20 MSE as the checkpointing and early-stopping criterion; lower is better",
    "越小越好 -> 取负号后越大越好": "lower is better, so the negative value is maximized",
    "发现更优模型": "Found an improved model",
    "已保存": "saved",
    "每个 epoch 保存 checkpoint，并仅保留最近 N 个": "Save a checkpoint after every epoch and retain only the most recent N",
    "早停触发 (核心基因拟合已达瓶颈)": "Early stopping triggered because fitting of key response genes has plateaued",
    "最终测试集评估 (使用保存的最佳权重)": "Final test-set evaluation using the best saved weights",
    "正在进行最终测试集 (Test Set) 评估...": "Running final test-set evaluation...",
    "使用 EMA 权重进行最终测试评估": "Use EMA weights for final test evaluation",
    "药物特征处理 (测试集)": "Process drug features for the test set",
    "最终评估结果 (Test Set)": "Final evaluation results (test set)",
    "你应该使用按 adata.var_names 对齐的 2000HVG embedding。": "Use a 2,000-HVG embedding aligned to adata.var_names.",
    "检测到 SCI-Plex 格式，正在适配列名...": "Detected SCI-Plex format; adapting column names...",
    "检测到 Adamson 格式，正在适配列名...": "Detected Adamson format; adapting column names...",
    "Adamson 格式清洗示例": "Example cleaned Adamson labels",
    "仅清理后缀噪声 (如 +ctrl / +control)": "remove suffix noise only, such as +ctrl or +control",
    "解析双扰动标签，避免塌缩成 'double'": "parse double-perturbation labels without collapsing them to 'double'",
    "解析双/三/多基因组合扰动标签": "parse double, triple, and higher-order combinatorial perturbation labels",
    "保留原始 perturbation 字符串，不做下划线截断清洗。": "retain the original perturbation string without underscore-based truncation.",
    "含 PAD": "including PAD",
    "已写入 adata.obs": "Written to adata.obs",
    "检测到 SMILES 信息，正在提取药物化学特征 (Morgan Fingerprint)...": "Detected SMILES data; extracting Morgan fingerprint drug features...",
    "警告: 无法解析 SMILES": "Warning: unable to parse SMILES",
    "药物特征提取完成，维度": "Drug-feature extraction complete; shape",
    "检测到剂量信息，正在处理 Dose 特征...": "Detected dose information; processing dose features...",
    "检测到 ATAC 特征": "Detected ATAC features",
    "ATAC 维度": "ATAC shape",
    "ATAC bank 不存在": "ATAC bank does not exist",
    "从 atac_bank 加载 ATAC 特征": "Loading ATAC features from atac_bank",
    "与当前 h5ad.var_names 不一致，请先按同一基因顺序构建 atac_bank。": "does not match the current h5ad.var_names. Rebuild atac_bank using the same gene order.",
    "中未找到背景向量键（除 'genes' 外）。": "does not contain a background-vector key other than 'genes'.",
    "使用 obs": "Using obs",
    "映射 ATAC 背景。": "to map ATAC backgrounds.",
    "未找到 obs": "obs field not found",
    "回退使用 obs": "falling back to obs",
    "以下背景在 atac_bank 中缺失，已用全0向量替代": "The following backgrounds are missing from atac_bank and were replaced with all-zero vectors",
    "从 atac_bank 构建样本级 ATAC 完成，维度": "Sample-level ATAC construction from atac_bank is complete; shape",
    "正在加载数据": "Loading data",
    "正在计算 control context 基线表达谱...": "Computing control-context baseline expression profiles...",
    "细胞系": "Cell line",
    "缺失控制组数据": "has no control data",
    "数据加载完成": "Data loading complete",
    "细胞": "cells",
    "基因": "genes",
    "尚未初始化，请先调用 load_data()。": "has not been initialized; call load_data() first.",
    "缺少自定义划分列": "is missing the custom split column",
    "采用自定义划分策略": "Using the custom split strategy",
    "划分结果": "Split sizes",
    "中 train/val/test 至少有一个为空。": "contains an empty train, validation, or test partition.",
    "采用按扰动基因划分策略 (Zero-shot 分层模式)...": "Using a perturbation-gene split with zero-shot stratification...",
    "训练集": "training set",
    "验证集": "validation set",
    "测试集": "test set",
    "采用随机划分策略...": "Using a random split...",
    "对于 perturbation zero-shot，val/test 通常没有 control。": "Validation and test splits usually contain no controls under perturbation zero-shot evaluation.",
    "因此统一复用 train control 作为 reference control bank。": "The training controls are therefore reused as the reference control bank.",
    "未找到 control 样本，无法构建 control pool。": "No control samples were found, so the control pool cannot be constructed.",
    "当前 split 内不存在可用 control 样本（global 模式）。": "The current split contains no usable control samples in global mode.",
    "当前 split 内不存在可用 control 样本，无法为非-control 样本匹配输入 control。": "The current split contains no usable controls for matching non-control samples.",
    "未找到可用于构造 control prototype 的候选 control。": "No candidate controls were found for constructing the control prototype.",
    "单细胞扰动数据集": "Single-cell perturbation dataset",
    "表达矩阵": "expression matrix",
    "标签 (1=匹配, 0=不匹配)": "labels (1=matched, 0=mismatched)",
    "确保是长整型用于 Embedding": "Ensure an integer type suitable for embedding lookup",
    "负责加载外部预训练基因向量并与当前数据的扰动 ID 对齐": "Load external pretrained gene vectors and align them to perturbation IDs in the current dataset",
    "预训练文件": "Pretrained file",
    "不存在。将使用随机初始化。": "does not exist. Random initialization will be used.",
    "正在加载预训练向量": "Loading pretrained vectors",
    "假设文件是 CSV (gene_name, dim1, dim2...)，无 header 或 index 为基因名": "Assume a CSV file with gene_name and feature columns, without a header or with gene names as the index",
    "根据实际下载格式可能需要微调 (例如 gene2vec 通常是 txt 或 bin)": "Adjust the parser when necessary for the downloaded format; gene2vec files are commonly txt or bin",
    "格式通常是: 第一行是 (n_genes, dim)，或者直接每行 gene val1 val2...": "files commonly contain (n_genes, dim) on the first line or one gene and its values per line",
    "我们直接使用 pd.read_csv 并根据内容自动判断": "Use pd.read_csv and infer the layout from the content",
    "如果第一行是元数据 (例如 24447 200)，我们需要过滤掉": "Remove the first line when it contains metadata such as 24447 200",
    "检测到文件首行包含元数据，已跳过。": "Detected and skipped metadata in the first line.",
    "已读取": "Loaded",
    "个基因的预训练向量，维度": "pretrained gene vectors; dimension",
    "构建权重矩阵": "Construct the weight matrix",
    "遍历我们数据中的所有扰动 ID": "Iterate over all perturbation IDs in the dataset",
    "Control 可以初始化为全 0 或特定向量": "Control can be initialized with zeros or a dedicated vector",
    "尝试匹配 (处理大小写不一致等问题)": "Attempt matching while handling capitalization differences",
    "预训练向量匹配成功率": "Pretrained-vector matching rate",
    "加载预训练向量失败": "Failed to load pretrained vectors",
    "生成模拟的单细胞扰动数据。": "Generate simulated single-cell perturbation data.",
    "逻辑：": "Procedure:",
    "为每个细胞随机分配一个 Cell Type。": "Randomly assign a cell type to every cell.",
    "生成基础表达矩阵 (HVGs)。": "Generate a baseline expression matrix of HVGs.",
    "生成正样本 (Label=1): 扰动与细胞类型匹配的真实生物学反应。": "Generate positive samples (Label=1) representing perturbations matched to cellular context.",
    "生成负样本 (Label=0): 随机替换扰动，模拟不匹配的情况。": "Generate negative samples (Label=0) by randomly replacing perturbations to create mismatched cases.",
    "随机分配细胞类型": "Randomly assign cell types",
    "生成基础表达矩阵 (模拟 HVGs)": "Generate the baseline expression matrix (simulated HVGs)",
    "为不同 Cell Type 生成不同的基础表达分布": "Generate distinct baseline expression distributions for different cell types",
    "每个细胞类型有一个独特的“特征均值向量”": "Each cell type has a distinct feature-mean vector",
    "定义扰动": "Define perturbations",
    "生成标签 (Label)": "Generate labels",
    "模拟某种规律：只有当 (perturbation_id + cell_type) 是偶数时，才产生反应 (Label=1)": "Simulate a rule in which a response occurs only when perturbation_id + cell_type is even (Label=1)",
    "这是一个简单的非线性规律，供 MLP 学习": "This is a simple nonlinear rule for the MLP to learn",
    "为了增加难度，给标签加一点噪音": "Add label noise to make the task more challenging",
    "翻转 5% 的标签": "Flip 5% of labels",
    "转换为 DataFrame 或 Tensor 格式": "Convert to DataFrame or tensor format",
    "将数据划分为训练集和验证集，并转换为 PyTorch Tensor": "Split data into training and validation sets and convert them to PyTorch tensors",
    "划分数据集 (手动实现)": "Split the dataset manually",
    "当前请求的可视化基因": "Requested visualization gene",
    "加载 Checkpoint": "Load checkpoint",
    "正在加载模型": "Loading model",
    "检测到 EMA 权重，优先使用 EMA 权重进行可视化评估": "Detected EMA weights; using them for visualization evaluation",
    "还原模型 (V9 架构)": "Restore the model (V9 architecture)",
    "执行推理并按扰动分组": "Run inference and group results by perturbation",
    "正在进行测试集推理评估...": "Running test-set inference evaluation...",
    "计算指标与准备绘图数据": "Compute metrics and prepare plotting data",
    "正在计算多维度评估指标...": "Computing multidimensional evaluation metrics...",
    "用户指定的 ROC 基因列表": "User-specified ROC gene list",
    "柱状图数据准备 (针对指定的基因)": "Prepare bar-plot data for the selected gene",
    "找到真实变化最大的 Top 20": "Select the top 20 genes with the largest observed changes",
    "计算 Top 20 重叠率": "Compute Top20 overlap",
    "构建“长表”数据 (Tidy Data)": "Construct a tidy long-format table",
    "新增: 打印详细的测试指标表格": "Added: print a detailed table of test metrics",
    "正在为": "Computing AUC for",
    "个测试扰动计算 AUC...": "test perturbations...",
    "AUC 计算完成": "AUC calculation complete",
    "格式化输出表格 (类似于用户提供的格式)": "Format the output table",
    "平均值": "Mean",
    "保存测试集基因列表": "Save the test-set gene list",
    "已将": "Saved",
    "个测试集基因保存至": "test-set genes to",
    "如果没找到指定的基因，自动选择测试集中 AUC 最高的基因作为备份": "If the requested gene is unavailable, select the test gene with the highest AUC as a fallback",
    "不在测试集中。": "is not in the test set.",
    "自动选择测试集表现最佳基因进行展示": "Automatically selected the best-performing test gene for display",
    "绘图 (2x2 布局)": "Plotting (2x2 layout)",
    "AUC 分布直方图": "AUC distribution histogram",
    "多线 ROC 曲线 (用户指定基因)": "Multi-line ROC curves for user-selected genes",
    "Top-20 DE 柱状图 (改为热图)": "Top20 DE bar plot converted to a heatmap",
    "转换数据格式为 Heatmap 所需的 Matrix": "Convert the data to the matrix format required by the heatmap",
    "结构": "structure",
    "提取 Control 基线": "Extract the control baseline",
    "注意: set_index 后可能有重复的 Gene (因为 df_p 是 long format)，需要先 filter": "Note: duplicate genes may remain after set_index because df_p is in long format, so filter first",
    "计算 Delta (Real - Ctrl, Pred - Ctrl) 并取绝对值": "Compute absolute deltas for observed minus control and predicted minus control",
    "确保索引对齐": "Ensure aligned indices",
    "合并为 DataFrame": "Combine into a DataFrame",
    "排序：按 Real_Delta 降序排列": "Sort by Real_Delta in descending order",
    "绘制热图 (使用 YlGnBu 颜色，类似提供的示例)": "Draw the heatmap using the YlGnBu palette",
    "移除 X 轴标签": "Remove the x-axis label",
    "性能汇总统计": "Performance summary statistics",
    "新增: 绘制第二个基因的柱状图 (替代纯文本区域的一部分)": "Added: draw a bar plot for a second gene instead of part of the text-only area",
    "我们在 axes[1, 1] 上方绘制柱状图，下方放统计文本": "Place the bar plot at the top of axes[1, 1] and summary text below it",
    "分割 axes[1, 1] 区域": "Divide the axes[1, 1] region",
    "准备第二个基因的数据 (硬编码为 AARS，或者如果 args.heatmap_gene 不是 AARS 则展示 AARS)": "Prepare data for a second gene, using AARS unless it is already the requested gene",
    "如果用户已经选了 AARS，那我们选个别的，比如 AUC 最高的": "If AARS is already selected, choose another gene such as the one with the highest AUC",
    "获取第二个基因的数据": "Retrieve data for the second gene",
    "注意：需要检查 second_gene 是否在测试集中": "Check whether second_gene is present in the test set",
    "模糊匹配": "Approximate match",
    "将文本放在 axes[1, 1] 的底部": "Place the text at the bottom of axes[1, 1]",
    "最终专业评估报告已生成": "Final evaluation report generated",
    "可视化组合扰动时默认复用同一 cell-line baseline ATAC，未显式建模组合特异 ATAC 变化。": "Combinatorial visualization reuses the same cell-line baseline ATAC profile and does not explicitly model combination-specific ATAC changes.",
    "图已保存": "Figure saved to",
    "基因报告": "Gene report",
}

PUNCTUATION = str.maketrans({
    "，": ",",
    "。": ".",
    "：": ":",
    "；": ";",
    "！": "!",
    "？": "?",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
})


def tracked_paths() -> list[pathlib.Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    return [pathlib.Path(name) for name in names if name]


def main() -> None:
    old_plan = pathlib.Path(".trae/documents/\u5355\u7ec6\u80de\u6270\u52a8\u7279\u5f02\u6027\u5efa\u6a21\u6846\u67b6\u642d\u5efa\u65b9\u6848.md")
    new_plan = pathlib.Path(".trae/documents/single_cell_perturbation_specificity_framework_plan.md")
    if old_plan.exists():
        old_plan.rename(new_plan)

    pathlib.Path("README.md").write_text(README, encoding="utf-8")
    pathlib.Path("docs/diffusion_methodology.md").write_text(METHODOLOGY, encoding="utf-8")
    new_plan.write_text(PLAN, encoding="utf-8")

    excluded = {
        pathlib.Path("scripts/translate_chinese_to_english.py"),
        pathlib.Path(".github/workflows/scan-chinese.yml"),
        pathlib.Path("CHINESE_SCAN.txt"),
    }

    paths = tracked_paths()
    if new_plan not in paths:
        paths.append(new_plan)

    for path in paths:
        if path in excluded or not path.exists() or path.suffix not in {".py", ".md", ".txt", ".yml", ".yaml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for source, target in REPLACEMENTS.items():
            text = text.replace(source, target)
        text = text.translate(PUNCTUATION)
        path.write_text(text, encoding="utf-8")

    remaining: list[str] = []
    for path in paths:
        if path in excluded or not path.exists():
            continue
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                remaining.append(f"{path}:{line_no}:{line}")

    pathlib.Path("CHINESE_SCAN.txt").write_text(
        "\n".join(remaining) + ("\n" if remaining else "") + f"TOTAL_MATCHING_LINES={len(remaining)}\n",
        encoding="utf-8",
    )
    if remaining:
        raise SystemExit(f"{len(remaining)} CJK-containing lines remain")


if __name__ == "__main__":
    main()
