
import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DataProcessor:
    def __init__(
        self,
        h5ad_path,
        test_size=0.1,
        val_size=0.1,
        split_strategy='random',
        split_col: str = 'split',
        perturb_parse_mode: str = 'raw',
        task_mode: str = 'single_gene',
        perturb_vocab_path: Optional[str] = None,
        atac_key: Optional[str] = None,
        atac_bank_path: Optional[str] = None,
        background_key: str = 'cell_context',
    ):
        self.h5ad_path = h5ad_path
        self.test_size = test_size
        self.val_size = val_size
        self.split_strategy = split_strategy
        self.split_col = split_col
        self.perturb_parse_mode = perturb_parse_mode
        task_mode = str(task_mode).strip().lower()
        if task_mode not in {'single_gene', 'translation', 'drug'}:
            raise ValueError(f"task_mode must be 'single_gene', 'translation' or 'drug', got {task_mode!r}")
        self.task_mode = task_mode
        self.perturb_vocab_path = perturb_vocab_path
        self.atac_key = atac_key
        self.atac_bank_path = atac_bank_path
        self.background_key = background_key

        self.adata = None
        self.perturb_map = None
        self.id_to_perturb = None
        self.perturb_gene_vocab = None
        self.perturb_gene_to_idx = None
        self.idx_to_perturb_gene = None
        self.pad_gene_token = "__PAD__"
        self.max_perturb_genes = 1
        self.perturb_gene_idx_matrix = None
        self.cell_line_map = None
        self.gene_to_idx = None
        self.condition_map = None
        self.condition_categories = None
        self.n_conditions = 0

        self.atac_features = None
        self.atac_dim = 0
        self.cell_line_atac_baselines = {}
        self.doses = None
        self.drug_embeddings = None

    def _ensure_dense_numpy(self, x):
        if issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def _canonicalize_perturbation_name(self, name: str) -> str:
        """Return a stable condition name for single/double/triple perturbations.

        Adamson UPR labels in scPerturb look like ATF6_PERK_IRE1_pMJ158.
        The model should see this as the gene set ATF6+EIF2AK3+ERN1, not as
        one unseen token. Keeping this parser here avoids yet another tiny
        preprocessing script wandering around the project like a lost sock.
        """
        x = str(name).strip()
        if x in {"", "nan", "None"}:
            return x

        ctrl_aliases = {
            "CTRL", "CTRL1", "ctrl", "Control", "control",
            "vehicle", "Vehicle", "mock", "Mock", "NTC", "ntc",
            "non-targeting", "non_targeting", "scramble", "scrambled",
            "3x_neg_ctrl_pMJ144-1", "3x_neg_ctrl_pMJ144-2",
        }
        if x in ctrl_aliases:
            return "control"

        adamson_map = {
            "ATF6_only_pMJ145": "ATF6",
            "PERK_only_pMJ146": "EIF2AK3",
            "IRE1_only_pMJ148": "ERN1",
            "ATF6_PERK_pMJ150": "ATF6+EIF2AK3",
            "ATF6_IRE1_pMJ152": "ATF6+ERN1",
            "PERK_IRE1_pMJ154": "EIF2AK3+ERN1",
            "ATF6_PERK_IRE1_pMJ158": "ATF6+EIF2AK3+ERN1",
        }
        if x in adamson_map:
            return adamson_map[x]

        # Common suffix cleanup from older Adamson/Norman processed objects.
        if x == "ctrl":
            return "control"
        if "+ctrl" in x:
            x = x.replace("+ctrl", "")
        if "+control" in x:
            x = x.replace("+control", "")

        # Normalize known aliases if a user already supplied a plus-separated condition.
        alias = {"PERK": "EIF2AK3", "IRE1": "ERN1", "IRE1A": "ERN1", "IRE1α": "ERN1"}
        if "+" in x:
            genes = [alias.get(g.strip(), g.strip()) for g in x.split("+") if g.strip()]
            return "+".join(genes) if genes else x

        return alias.get(x, x)

    def _split_perturbation_genes(self, name: str):
        name = self._canonicalize_perturbation_name(name)
        if name == "control":
            return []
        if "+" in name:
            return [g for g in name.split("+") if g]
        return [name]

    def _prepare_metadata(self):
        if 'product_name' in self.adata.obs and 'perturbation' not in self.adata.obs:
            print(">>> 检测到 SCI-Plex 格式，正在适配列名...")
            self.adata.obs['perturbation'] = self.adata.obs['product_name'].astype(str)
            self.adata.obs.loc[self.adata.obs['perturbation'].isin(['Vehicle', 'control']), 'perturbation'] = 'control'

        # no-cell-line cross-species mode:
        # Do not auto-create obs['cell_line'] from obs['cell_type'].

        if 'condition' in self.adata.obs and 'perturbation' not in self.adata.obs:
            print(">>> 检测到 Adamson 格式，正在适配列名...")
            self.adata.obs['perturbation'] = self.adata.obs['condition'].astype(str)

            def clean_adamson_pert(x):
                if x == 'ctrl':
                    return 'control'
                if '+ctrl' in x:
                    return x.split('+')[0]
                return x

            self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(clean_adamson_pert)
            print(f">>> Adamson 格式清洗示例: {self.adata.obs['perturbation'].unique()[:5]}")

        if 'SMILES' in self.adata.obs and 'smiles' not in self.adata.obs:
            self.adata.obs['smiles'] = self.adata.obs['SMILES']

        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype(str)
        ctrl_aliases = {"CTRL", "CTRL1", "ctrl", "Control", "vehicle", "Vehicle"}
        self.adata.obs.loc[self.adata.obs["perturbation"].isin(ctrl_aliases), "perturbation"] = "control"

        if self.perturb_parse_mode == 'single_gene_suffix_clean':
            print(">>> perturb_parse_mode=single_gene_suffix_clean: 清理 +ctrl/+control 后缀")
            self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(
                lambda x: str(x).replace('+control', '').replace('+ctrl', '') if str(x) != 'control' else 'control'
            )
        elif self.perturb_parse_mode == 'double_gene_parse':
            print(">>> perturb_parse_mode=double_gene_parse: 解析 double_* 双扰动标签")

            def _parse_double(x):
                x = str(x)
                if x == 'control':
                    return x
                if x.startswith('double_'):
                    parts = [p for p in x.split('_')[1:] if p]
                    if len(parts) >= 2:
                        a, b = sorted(parts[:2])
                        return f"{a}+{b}"
                return x

            self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(_parse_double)
        elif self.perturb_parse_mode == 'adamson_upr':
            print(">>> perturb_parse_mode=adamson_upr: 强制解析 Adamson UPR single/double/triple 标签")
        else:
            print(">>> perturb_parse_mode=raw: 保留字符串，但自动识别 Adamson UPR 和 plus-separated 多基因扰动。")

        # Canonical condition labels used by the diffusion model.
        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(self._canonicalize_perturbation_name)
        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype('category')
        self.perturb_categories = self.adata.obs['perturbation'].cat.categories.tolist()
        self.perturb_map = {name: i for i, name in enumerate(self.perturb_categories)}
        self.id_to_perturb = {i: name for name, i in self.perturb_map.items()}

        # Build a component-gene vocabulary rather than treating a combination as one token.
        # PAD is fixed to index 0 because the model masks index 0 when averaging multi-gene embeddings.
        component_gene_set = set()
        perturb_gene_lists = []
        for name in self.adata.obs['perturbation'].astype(str).values:
            genes = self._split_perturbation_genes(name)
            perturb_gene_lists.append(genes)
            component_gene_set.update(genes)

        if self.perturb_vocab_path is not None:
            if not os.path.exists(self.perturb_vocab_path):
                raise FileNotFoundError(f"perturb vocab file not found: {self.perturb_vocab_path}")
            with open(self.perturb_vocab_path, 'r', encoding='utf-8') as f:
                for ln in f:
                    g = ln.strip()
                    if not g or g == 'control':
                        continue
                    component_gene_set.add(g)

        self.perturb_gene_vocab = [self.pad_gene_token] + sorted(component_gene_set)
        self.perturb_gene_to_idx = {g: i for i, g in enumerate(self.perturb_gene_vocab)}
        self.idx_to_perturb_gene = {i: g for g, i in self.perturb_gene_to_idx.items()}
        self.max_perturb_genes = max(1, max((len(g) for g in perturb_gene_lists), default=0))
        print(
            f">>> component perturb gene vocab size: {len(self.perturb_gene_vocab)} "
            f"(PAD=0, max_genes_per_condition={self.max_perturb_genes})"
        )

        pad_idx = self.perturb_gene_to_idx[self.pad_gene_token]
        perturb_gene_idx_matrix, is_control = [], []
        for genes in perturb_gene_lists:
            if len(genes) == 0:
                perturb_gene_idx_matrix.append([pad_idx] * self.max_perturb_genes)
                is_control.append(1)
            else:
                idxs = [self.perturb_gene_to_idx[g] for g in genes]
                idxs = idxs + [pad_idx] * (self.max_perturb_genes - len(idxs))
                perturb_gene_idx_matrix.append(idxs)
                is_control.append(0)

        self.perturb_gene_idx_matrix = np.asarray(perturb_gene_idx_matrix, dtype=np.int64)
        self.adata.obs['is_control'] = np.array(is_control, dtype=np.int64)
        # Keep a legacy scalar column for old notebooks; loaders use the full matrix above.
        self.adata.obs['perturb_gene_idx'] = self.perturb_gene_idx_matrix[:, 0].astype(np.int64)
        self.adata.obs['nperts_clean'] = [0 if v else int(np.sum(row != pad_idx)) for row, v in zip(self.perturb_gene_idx_matrix, is_control)]
        self.adata.obs['perturb_gene_set'] = [
            '+'.join(genes) if genes else 'control'
            for genes in perturb_gene_lists
        ]
        print(">>> 已写入: perturb_gene_idx_matrix / is_control / nperts_clean / perturb_gene_set")

        if self.task_mode == 'translation':
            cond_col = 'condition' if 'condition' in self.adata.obs else 'perturbation'
            self.adata.obs[cond_col] = self.adata.obs[cond_col].astype('category')
            self.condition_categories = self.adata.obs[cond_col].cat.categories.tolist()
            self.condition_map = {name: i for i, name in enumerate(self.condition_categories)}
            self.n_conditions = len(self.condition_categories)
            self.adata.obs['condition_id'] = self.adata.obs[cond_col].cat.codes.astype(np.int64)
            if 'orig_condition' in self.adata.obs:
                self.adata.obs['source_flag'] = (
                    self.adata.obs[cond_col].astype(str).values == self.adata.obs['orig_condition'].astype(str).values
                ).astype(np.int64)
            else:
                self.adata.obs['source_flag'] = np.zeros(self.adata.n_obs, dtype=np.int64)
            print(f">>> translation mode condition vocab size: {self.n_conditions} (col={cond_col})")
        else:
            self.adata.obs['condition_id'] = np.zeros(self.adata.n_obs, dtype=np.int64)
            self.adata.obs['source_flag'] = np.zeros(self.adata.n_obs, dtype=np.int64)
            self.n_conditions = 0

        if self.background_key in self.adata.obs:
            context_col = self.background_key
        elif "cell_context" in self.adata.obs:
            context_col = "cell_context"
        elif "condition" in self.adata.obs:
            context_col = "condition"
        elif "species" in self.adata.obs:
            context_col = "species"
        elif "source_batch" in self.adata.obs:
            context_col = "source_batch"
        else:
            context_col = "_global_context"
            self.adata.obs[context_col] = "global"

        self.cell_line_col = context_col  # legacy name, now used as context id
        self.adata.obs[context_col] = self.adata.obs[context_col].astype('category')
        self.cell_line_categories = self.adata.obs[context_col].cat.categories.tolist()
        self.cell_line_map = {name: i for i, name in enumerate(self.cell_line_categories)}
        print(f">>> no-cell-line mode: using obs['{context_col}'] as control-pool context, not as cell-line embedding.")


    def _prepare_drug_features(self):
        self.drug_embeddings = None
        if 'smiles' not in self.adata.obs:
            return

        print(">>> 检测到 SMILES 信息，正在提取药物化学特征 (Morgan Fingerprint)...")
        unique_perts = self.adata.obs[['perturbation', 'smiles']].drop_duplicates('perturbation')
        pert_to_smiles = dict(zip(unique_perts['perturbation'], unique_perts['smiles']))

        drug_feats = []
        for name in self.perturb_categories:
            if name == 'control' or name not in pert_to_smiles:
                drug_feats.append(np.zeros(2048, dtype=np.float32))
                continue

            mol = Chem.MolFromSmiles(pert_to_smiles[name])
            if mol is None:
                print(f"!!! 警告: 无法解析 SMILES: {name}")
                drug_feats.append(np.zeros(2048, dtype=np.float32))
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            drug_feats.append(np.asarray(fp, dtype=np.float32))

        self.drug_embeddings = torch.tensor(np.stack(drug_feats, axis=0), dtype=torch.float32)
        print(f">>> 药物特征提取完成，维度: {self.drug_embeddings.shape}")

    def _prepare_dose_features(self):
        if 'dose' not in self.adata.obs:
            self.doses = None
            return

        print(">>> 检测到剂量信息，正在处理 Dose 特征...")
        doses = self.adata.obs['dose'].values.astype(np.float32)
        doses = np.log1p(doses)
        d_min, d_max = doses.min(), doses.max()
        if d_max > d_min:
            doses = (doses - d_min) / (d_max - d_min)
        else:
            doses = np.zeros_like(doses)
        self.doses = torch.tensor(doses, dtype=torch.float32)

    def _prepare_atac_features(self, atac_key=None, atac_bank_path=None, background_key='cell_context'):
        if self.atac_features is not None:
            return

        self.atac_features = None
        candidate_keys = [atac_key] if atac_key is not None else ['X_atac', 'atac', 'atac_feat']
        candidate_keys = [k for k in candidate_keys if k is not None]

        for key in candidate_keys:
            if key in self.adata.obsm:
                print(f">>> 检测到 ATAC 特征: obsm['{key}']")
                atac = self._ensure_dense_numpy(self.adata.obsm[key]).astype(np.float32)
                self.atac_features = torch.tensor(atac, dtype=torch.float32)
                self.atac_dim = int(self.atac_features.shape[1])
                print(f">>> ATAC 维度: {self.atac_features.shape}")
                break

        if self.atac_features is None and atac_bank_path is not None:
            if not os.path.exists(atac_bank_path):
                raise FileNotFoundError(f"ATAC bank 不存在: {atac_bank_path}")

            print(f">>> 从 atac_bank 加载 ATAC 特征: {atac_bank_path}")
            bank = np.load(atac_bank_path, allow_pickle=True)

            if 'genes' in bank:
                bank_genes = [str(x) for x in bank['genes'].tolist()]
                adata_genes = [str(x) for x in self.adata.var_names.tolist()]
                if len(bank_genes) != len(adata_genes) or any(g1 != g2 for g1, g2 in zip(bank_genes, adata_genes)):
                    raise ValueError("atac_bank['genes'] 与当前 h5ad.var_names 不一致，请先按同一基因顺序构建 atac_bank。")

            bank_map = {k: bank[k].astype(np.float32) for k in bank.files if k != 'genes'}
            if len(bank_map) == 0:
                raise ValueError("atac_bank 中未找到背景向量键（除 'genes' 外）。")

            if background_key in self.adata.obs:
                bg_values = self.adata.obs[background_key].astype(str).values
                print(f">>> 使用 obs['{background_key}'] 映射 ATAC 背景。")
            else:
                bg_values = self.adata.obs[self.cell_line_col].astype(str).values
                print(f">>> 未找到 obs['{background_key}']，回退使用 obs['{self.cell_line_col}']。")

            sample_vec = next(iter(bank_map.values()))
            atac_arr = np.zeros((self.adata.n_obs, sample_vec.shape[0]), dtype=np.float32)
            missing_bg = set()
            for i, bg in enumerate(bg_values):
                if bg in bank_map:
                    atac_arr[i] = bank_map[bg]
                else:
                    missing_bg.add(bg)

            if len(missing_bg) > 0:
                print(f"!!! 警告: 以下背景在 atac_bank 中缺失，已用全0向量替代: {sorted(list(missing_bg))[:10]}")

            self.atac_features = torch.tensor(atac_arr, dtype=torch.float32)
            self.atac_dim = int(self.atac_features.shape[1])
            print(f">>> 从 atac_bank 构建样本级 ATAC 完成，维度: {self.atac_features.shape}")

        if self.atac_features is not None:
            self.cell_line_atac_baselines = {}
            cell_line_ids = self.adata.obs[self.cell_line_col].cat.codes.values
            atac_np = self.atac_features.numpy()
            for cl_id in np.unique(cell_line_ids):
                idx = np.where(cell_line_ids == cl_id)[0]
                if len(idx) == 0:
                    continue
                self.cell_line_atac_baselines[int(cl_id)] = torch.tensor(atac_np[idx].mean(axis=0), dtype=torch.float32)

    def get_cell_line_control(self, cell_line_idx: int, device=None):
        x = self.cell_line_baselines[cell_line_idx]
        if device is not None:
            x = x.to(device)
        return x

    def get_cell_line_atac(self, cell_line_idx: int, device=None):
        if not self.cell_line_atac_baselines or cell_line_idx not in self.cell_line_atac_baselines:
            return None
        x = self.cell_line_atac_baselines[cell_line_idx]
        if device is not None:
            x = x.to(device)
        return x

    def load_data(self):
        print(f">>> 正在加载数据: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)

        self._prepare_metadata()
        self._prepare_drug_features()
        self._prepare_dose_features()
        self._prepare_atac_features(
            atac_key=self.atac_key,
            atac_bank_path=self.atac_bank_path,
            background_key=self.background_key,
        )

        print(">>> 正在计算 control context 基线表达谱...")
        self.cell_line_baselines = {}
        cell_line_values = self.adata.obs[self.cell_line_col].values
        pert_values = self.adata.obs['perturbation'].values
        X = self.adata.X

        for cl_name, cl_id in self.cell_line_map.items():
            ctrl_mask = (cell_line_values == cl_name) & (pert_values == 'control')
            ctrl_idx = np.where(ctrl_mask)[0]
            if len(ctrl_idx) > 0:
                avg_expr = X[ctrl_idx].mean(axis=0)
                if issparse(avg_expr):
                    avg_expr = avg_expr.toarray()
                avg_expr = np.asarray(avg_expr).flatten()
                self.cell_line_baselines[cl_id] = torch.tensor(avg_expr, dtype=torch.float32)
            else:
                print(f"!!! 警告: 细胞系 {cl_name} 缺失控制组数据")

        self.gene_to_idx = {gene: i for i, gene in enumerate(self.adata.var_names)}
        print(f">>> 数据加载完成: {self.adata.n_obs} 细胞, {self.adata.n_vars} 基因")
        return self.adata.n_vars, len(self.perturb_categories), len(self.cell_line_categories)

    def encode_structured_perturbation_names(self, perturb_names):
        if self.perturb_gene_to_idx is None:
            raise ValueError("perturb_gene vocab 尚未初始化，请先调用 load_data()。")

        pad_idx = self.perturb_gene_to_idx[self.pad_gene_token]
        rows, p_is_control = [], []
        for p_name in perturb_names:
            name = self._canonicalize_perturbation_name(str(p_name))
            genes = self._split_perturbation_genes(name)
            if len(genes) == 0:
                rows.append([pad_idx] * self.max_perturb_genes)
                p_is_control.append(1.0)
            else:
                missing = [g for g in genes if g not in self.perturb_gene_to_idx]
                if missing:
                    raise KeyError(f"扰动基因不在 perturb_gene_vocab 中: {missing}. 当前条件: {name}")
                idxs = [self.perturb_gene_to_idx[g] for g in genes]
                idxs = idxs + [pad_idx] * (self.max_perturb_genes - len(idxs))
                rows.append(idxs[: self.max_perturb_genes])
                p_is_control.append(0.0)

        return {
            'perturb_gene_idx': torch.tensor(rows, dtype=torch.long),
            'is_control': torch.tensor(p_is_control, dtype=torch.float32),
        }

    def prepare_loaders(
        self,
        batch_size=2048,
        rna_noise=0.1,
        gene_mask_rate=0.05,
        scale_rate=0.05,
        num_workers=0,
        atac_key=None,
        atac_bank_path=None,
        background_key='cell_context',
        control_match_mode='random',
        control_match_k=32,
        control_match_scope='cell_line',
        control_prototype_mode='single',
        control_prototype_temp=1.0,
        control_knn_cache_path=None,
    ):
        # Cached ATAC-KNN configuration
        control_knn_cache_path = (
            control_knn_cache_path
            or os.environ.get('SCERSO_CONTROL_KNN_CACHE')
        )
        num_workers = int(
            os.environ.get(
                'SCERSO_NUM_WORKERS',
                num_workers,
            )
        )

        if self.atac_features is None and (atac_key is not None or atac_bank_path is not None):
            self._prepare_atac_features(
                atac_key=atac_key,
                atac_bank_path=atac_bank_path,
                background_key=background_key,
            )

        X = self.adata.X
        perturb_ids = self.adata.obs['perturbation'].cat.codes.values
        perturb_gene_idx_ids = self.perturb_gene_idx_matrix.astype(np.int64)
        is_control_ids = self.adata.obs['is_control'].values.astype(np.float32)
        condition_id_ids = self.adata.obs['condition_id'].values.astype(np.int64)
        source_flag_ids = self.adata.obs['source_flag'].values.astype(np.int64)
        cell_line_ids = self.adata.obs[self.cell_line_col].cat.codes.values
        control_id = self.perturb_map.get('control', None)

        batch_ids = None
        if 'batch' in self.adata.obs:
            self.adata.obs['batch'] = self.adata.obs['batch'].astype('category')
            batch_ids = self.adata.obs['batch'].cat.codes.values

        indices = np.arange(self.adata.n_obs)

        if self.split_strategy == 'custom':
            if self.split_col not in self.adata.obs:
                raise ValueError(f"adata.obs 缺少自定义划分列: {self.split_col}")
            split_values = self.adata.obs[self.split_col].astype(str).values
            train_idx = np.where(split_values == 'train')[0]
            val_idx = np.where(split_values == 'val')[0]
            # Accept test/test_triple/heldout-like labels so older prepared h5ad files still work.
            test_idx = np.where(np.char.startswith(split_values.astype(str), 'test') | (split_values == 'heldout'))[0]
            if len(val_idx) == 0:
                # Some quick prepared files only have train/test_triple. Make a small validation set
                # from non-control training cells and keep all controls in train for the control pool.
                print(">>> 自定义 split 未发现 val，自动从 train 非 control 细胞中划出验证集。")
                train_non_ctrl = train_idx[self.adata.obs['perturbation'].astype(str).values[train_idx] != 'control']
                train_ctrl = train_idx[self.adata.obs['perturbation'].astype(str).values[train_idx] == 'control']
                if len(train_non_ctrl) > 1:
                    train_non_ctrl, val_idx = train_test_split(
                        train_non_ctrl,
                        test_size=max(self.val_size, 0.05),
                        random_state=42,
                    )
                    train_idx = np.concatenate([train_non_ctrl, train_ctrl])
            print(f">>> 采用自定义划分策略: obs['{self.split_col}']")
            print(f">>> 划分结果: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                raise ValueError(f"自定义划分列 {self.split_col} 中 train/val/test 至少有一个为空。")
        elif self.split_strategy == 'adamson_triple':
            print(">>> 采用 Adamson UPR 三基因 held-out 划分: train=control+single+double, test=ATF6+EIF2AK3+ERN1")
            triple = 'ATF6+EIF2AK3+ERN1'
            pert_names = self.adata.obs['perturbation'].astype(str).values
            test_idx = np.where(pert_names == triple)[0]
            ctrl_idx = np.where(pert_names == 'control')[0]
            train_val_idx = np.where((pert_names != triple) & (pert_names != 'control'))[0]
            if len(test_idx) == 0:
                raise ValueError("未找到 Adamson triple condition: ATF6+EIF2AK3+ERN1")
            if len(train_val_idx) <= 1:
                raise ValueError("可用于训练/验证的 single/double 扰动细胞太少。")
            train_non_ctrl, val_idx = train_test_split(
                train_val_idx,
                test_size=max(self.val_size, 0.05),
                random_state=42,
                stratify=self.adata.obs['perturbation'].astype(str).values[train_val_idx]
                if len(set(pert_names[train_val_idx])) > 1 else None,
            )
            train_idx = np.concatenate([train_non_ctrl, ctrl_idx])
            print(f">>> 划分结果: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        elif self.split_strategy == 'perturbation':
            print(">>> 采用按扰动基因划分策略 (Zero-shot 分层模式)...")
            real_perts = [p for p in self.perturb_categories if p != 'control']
            np.random.seed(42)
            np.random.shuffle(real_perts)

            n_test = int(len(real_perts) * self.test_size)
            n_val = int(len(real_perts) * self.val_size)

            test_p = set(real_perts[:n_test])
            val_p = set(real_perts[n_test:n_test + n_val])
            train_p = set(real_perts[n_test + n_val:])

            def get_idx(pert_set):
                mask = self.adata.obs['perturbation'].isin(pert_set)
                return np.where(mask)[0]

            train_idx_p = get_idx(train_p)
            val_idx = get_idx(val_p)
            test_idx = get_idx(test_p)
            ctrl_idx = np.where(self.adata.obs['perturbation'] == 'control')[0]
            train_idx = np.concatenate([train_idx_p, ctrl_idx])

            print(f">>> 划分结果: 训练集 {len(train_p)} 基因(+Control), 验证集 {len(val_p)} 基因, 测试集 {len(test_p)} 基因")
        else:
            print(">>> 采用随机划分策略...")
            train_idx, temp = train_test_split(indices, test_size=(self.val_size + self.test_size), random_state=42)
            val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=42)

        def build_control_pools(ctrl_indices):
            control_pool_coarse = {}
            control_pool_fine = {} if batch_ids is not None else None
            for gidx in ctrl_indices:
                c_id = int(cell_line_ids[gidx])
                control_pool_coarse.setdefault(c_id, []).append(int(gidx))
                if batch_ids is not None:
                    b_id = int(batch_ids[gidx])
                    control_pool_fine.setdefault((c_id, b_id), []).append(int(gidx))
            return control_pool_coarse, control_pool_fine

        all_ctrl_idx = np.where(perturb_ids == control_id)[0] if control_id is not None else np.array([], dtype=np.int64)
        train_ctrl_idx = np.intersect1d(train_idx, all_ctrl_idx)
        val_ctrl_idx = np.intersect1d(val_idx, all_ctrl_idx)
        test_ctrl_idx = np.intersect1d(test_idx, all_ctrl_idx)

        train_control_pool_coarse, train_control_pool_fine = build_control_pools(train_ctrl_idx)
        val_control_pool_coarse, val_control_pool_fine = build_control_pools(val_ctrl_idx)
        test_control_pool_coarse, test_control_pool_fine = build_control_pools(test_ctrl_idx)
        # 对于 perturbation zero-shot，val/test 通常没有 control。
        # 因此统一复用 train control 作为 reference control bank。
        ref_control_pool_coarse = train_control_pool_coarse
        ref_control_pool_fine = train_control_pool_fine if batch_ids is not None else None

        if len(all_ctrl_idx) == 0:
            raise ValueError("未找到 control 样本，无法构建 control pool。")

        # ---------------------------------------------
        # Optional precomputed ATAC-KNN cache
        # ---------------------------------------------
        knn_cache_indices = None
        knn_cache_weights = None

        if control_knn_cache_path:
            if not os.path.exists(control_knn_cache_path):
                raise FileNotFoundError(
                    f"ATAC-KNN cache not found: "
                    f"{control_knn_cache_path}"
                )

            with np.load(
                control_knn_cache_path,
                allow_pickle=False,
            ) as cache_data:
                if 'control_indices' not in cache_data:
                    raise KeyError(
                        "cache missing 'control_indices'"
                    )
                if 'control_weights' not in cache_data:
                    raise KeyError(
                        "cache missing 'control_weights'"
                    )

                knn_cache_indices = np.asarray(
                    cache_data['control_indices'],
                    dtype=np.int64,
                ).copy()
                knn_cache_weights = np.asarray(
                    cache_data['control_weights'],
                    dtype=np.float32,
                ).copy()

            expected_cells = self.adata.n_obs

            if knn_cache_indices.shape[0] != expected_cells:
                raise ValueError(
                    "KNN cache cell count mismatch: "
                    f"{knn_cache_indices.shape[0]} vs "
                    f"{expected_cells}"
                )

            if knn_cache_indices.shape != knn_cache_weights.shape:
                raise ValueError(
                    "KNN indices/weights shape mismatch"
                )

            if knn_cache_indices.shape[1] < int(control_match_k):
                raise ValueError(
                    "KNN cache contains fewer neighbors "
                    f"than control_match_k={control_match_k}"
                )

            knn_cache_indices = knn_cache_indices[
                :, :int(control_match_k)
            ]
            knn_cache_weights = knn_cache_weights[
                :, :int(control_match_k)
            ]

            print(
                ">>> Loaded cached ATAC-KNN: "
                f"{control_knn_cache_path}"
            )
            print(
                ">>> Cached neighbors shape: "
                f"{knn_cache_indices.shape}; "
                f"num_workers={num_workers}"
            )

        class GenerativeDataset(Dataset):
            def __init__(
                self,
                full_rna,
                full_atac,
                sample_indices,
                p_ids,
                c_ids,
                p_gene_idx,
                p_is_control,
                p_condition_id,
                p_source_flag,
                doses,
                atac_feats,
                control_id,
                control_pool_coarse,
                id_to_perturb=None,
                parse_structured_perturbation=None,
                control_pool_fine=None,
                local_batch_ids=None,
                rna_noise=0.0,
                gene_mask_rate=0.0,
                scale_rate=0.0,
                is_train=True,
                seed=42,
                control_match_mode='random',
                control_match_k=32,
                control_match_scope='cell_line',
                control_prototype_mode='single',
                control_prototype_temp=1.0,
                knn_cache_indices=None,
                knn_cache_weights=None,
            ):
                self.full_rna = full_rna
                self.full_atac = full_atac
                self.sample_indices = sample_indices
                self.p_ids = p_ids
                self.c_ids = c_ids
                self.p_gene_idx = p_gene_idx
                self.p_is_control = p_is_control
                self.p_condition_id = p_condition_id
                self.p_source_flag = p_source_flag
                self.doses = doses
                self.atac_feats = atac_feats
                self.control_id = control_id
                self.id_to_perturb = id_to_perturb
                self.parse_structured_perturbation = parse_structured_perturbation
                self.control_pool_coarse = control_pool_coarse
                self.control_pool_fine = control_pool_fine
                self.local_batch_ids = local_batch_ids
                self.rna_noise = rna_noise
                self.gene_mask_rate = gene_mask_rate
                self.scale_rate = scale_rate
                self.is_train = is_train
                self.rng = np.random.RandomState(seed)
                self.control_match_mode = control_match_mode
                self.control_match_k = max(int(control_match_k), 1)
                self.control_match_scope = control_match_scope
                self.control_prototype_mode = control_prototype_mode
                self.control_prototype_temp = max(float(control_prototype_temp), 1e-6)
                self.knn_cache_indices = knn_cache_indices
                self.knn_cache_weights = knn_cache_weights
                if len(self.control_pool_coarse) > 0:
                    self.global_control_fallback = np.concatenate(
                        [np.array(v, dtype=np.int64) for v in self.control_pool_coarse.values()]
                    )
                else:
                    self.global_control_fallback = np.array([], dtype=np.int64)
                self.fixed_ctrl_idx = []

                if not self.is_train:
                    for i in range(len(self.p_ids)):
                        p_id = int(self.p_ids[i])
                        if p_id == self.control_id:
                            self.fixed_ctrl_idx.append(None)
                            continue
                        self.fixed_ctrl_idx.append(None)

            def __len__(self):
                return len(self.p_ids)

            def __getitem__(self, idx):
                target_rna = self._get_rna_from_global(self.sample_indices[idx])
                c_id = int(self.c_ids[idx])
                p_id = int(self.p_ids[idx])
                perturb_gene_idx = self.p_gene_idx[idx]
                is_control = float(self.p_is_control[idx])
                condition_id = int(self.p_condition_id[idx])
                source_flag = int(self.p_source_flag[idx])

                dose_val = self.doses[idx] if self.doses is not None else torch.tensor(0.0)
                atac_val = self.atac_feats[idx] if self.atac_feats is not None else None

                if p_id == self.control_id:
                    input_rna = target_rna.clone()
                else:
                    if self.control_prototype_mode != 'single':
                        input_rna = self._build_control_prototype(idx, c_id)
                    else:
                        if self.is_train:
                            ctrl_gidx = self._sample_control_index(idx, c_id)
                        else:
                            if self.fixed_ctrl_idx[idx] is None:
                                self.fixed_ctrl_idx[idx] = self._sample_control_index(idx, c_id)
                            ctrl_gidx = self.fixed_ctrl_idx[idx]
                        input_rna = self._get_rna_from_global(ctrl_gidx)

                if self.is_train:
                    if self.rna_noise > 0:
                        target_rna = target_rna + torch.randn_like(target_rna) * self.rna_noise
                    if self.gene_mask_rate > 0:
                        mask = torch.rand_like(target_rna) > self.gene_mask_rate
                        target_rna = target_rna * mask

                item = {
                    'rna_control': input_rna,
                    'rna_target': target_rna,
                    'perturb': torch.tensor(p_id, dtype=torch.long),
                    'perturb_gene_idx': torch.as_tensor(perturb_gene_idx, dtype=torch.long),
                    'is_control': torch.tensor(is_control, dtype=torch.float32),
                    'condition_id': torch.tensor(condition_id, dtype=torch.long),
                    'source_flag': torch.tensor(source_flag, dtype=torch.long),
                    'context_id': torch.tensor(c_id, dtype=torch.long),
                    'dose': dose_val,
                }
                if atac_val is not None:
                    item['atac_feat'] = atac_val
                return item

            def _get_control_candidates(self, local_idx, c_id):
                if self.control_match_scope == 'global':
                    if self.global_control_fallback.size == 0:
                        raise ValueError("当前 split 内不存在可用 control 样本（global 模式）。")
                    return self.global_control_fallback
                if self.local_batch_ids is not None and self.control_pool_fine is not None:
                    b_id = int(self.local_batch_ids[local_idx])
                    key = (c_id, b_id)
                    if key in self.control_pool_fine and len(self.control_pool_fine[key]) > 0:
                        return self.control_pool_fine[key]
                if c_id in self.control_pool_coarse and len(self.control_pool_coarse[c_id]) > 0:
                    return self.control_pool_coarse[c_id]
                if self.global_control_fallback.size == 0:
                    raise ValueError("当前 split 内不存在可用 control 样本，无法为非-control 样本匹配输入 control。")
                return self.global_control_fallback

            def _sample_control_index(self, local_idx, c_id):
                ranked_candidates, ranked_dists = self._rank_control_candidates(local_idx, c_id)
                if ranked_dists is None:
                    return int(self.rng.choice(ranked_candidates))
                k = min(self.control_match_k, len(ranked_candidates))
                if self.is_train and k > 1:
                    return int(self.rng.choice(ranked_candidates[:k]))
                return int(ranked_candidates[0])

            def _rank_control_candidates(self, local_idx, c_id):
                candidates = self._get_control_candidates(local_idx, c_id)
                candidates = np.asarray(candidates, dtype=np.int64)
                if self.control_match_mode != 'atac_knn' or self.full_atac is None or len(candidates) <= 1:
                    return candidates.tolist(), None

                target_global_idx = int(self.sample_indices[local_idx])
                target_atac = self.full_atac[target_global_idx]
                if target_atac.dim() != 1:
                    target_atac = target_atac.view(-1)
                cand_idx_t = torch.as_tensor(candidates, dtype=torch.long)
                cand_atac = self.full_atac[cand_idx_t]
                dists = torch.sum((cand_atac - target_atac.unsqueeze(0)) ** 2, dim=1)
                order = torch.argsort(dists, dim=0).cpu().numpy()
                sorted_candidates = candidates[order].tolist()
                sorted_dists = dists[order].detach().cpu()
                return sorted_candidates, sorted_dists

            def _build_control_prototype(self, local_idx, c_id):
                # Fast path: use precomputed global control indices and weights.
                if (
                    self.knn_cache_indices is not None
                    and self.knn_cache_weights is not None
                ):
                    target_global_idx = int(
                        self.sample_indices[local_idx]
                    )

                    cached_indices = np.asarray(
                        self.knn_cache_indices[
                            target_global_idx
                        ],
                        dtype=np.int64,
                    )
                    cached_weights = np.asarray(
                        self.knn_cache_weights[
                            target_global_idx
                        ],
                        dtype=np.float32,
                    )

                    valid_indices = cached_indices >= 0
                    cached_indices = cached_indices[valid_indices]
                    cached_weights = cached_weights[valid_indices]

                    k = min(
                        self.control_match_k,
                        len(cached_indices),
                    )

                    cached_indices = cached_indices[:k]
                    cached_weights = cached_weights[:k]

                    if len(cached_indices) > 0:
                        if (
                            self.control_prototype_mode
                            == 'topk_mean'
                        ):
                            picked = cached_indices.tolist()
                            rnas = torch.stack(
                                [
                                    self._get_rna_from_global(
                                        int(gidx)
                                    )
                                    for gidx in picked
                                ],
                                dim=0,
                            )
                            return torch.mean(
                                rnas,
                                dim=0,
                            )

                        # Ignore exact zero weights.
                        positive = (
                            np.isfinite(cached_weights)
                            & (cached_weights > 0)
                        )

                        if not np.any(positive):
                            positive = np.zeros_like(
                                cached_weights,
                                dtype=bool,
                            )
                            positive[0] = True

                        picked = cached_indices[positive]
                        picked_weights = cached_weights[positive]

                        rnas = torch.stack(
                            [
                                self._get_rna_from_global(
                                    int(gidx)
                                )
                                for gidx in picked
                            ],
                            dim=0,
                        )

                        weights = torch.as_tensor(
                            picked_weights,
                            dtype=rnas.dtype,
                        )
                        weights = weights / torch.clamp(
                            weights.sum(),
                            min=1e-12,
                        )

                        return torch.sum(
                            weights.unsqueeze(1) * rnas,
                            dim=0,
                        )

                # Original fallback path.
                ranked_candidates, ranked_dists = self._rank_control_candidates(local_idx, c_id)
                if len(ranked_candidates) == 0:
                    raise ValueError("未找到可用于构造 control prototype 的候选 control。")

                k = min(self.control_match_k, len(ranked_candidates))
                if ranked_dists is None and self.is_train and len(ranked_candidates) > k:
                    picked = self.rng.choice(np.asarray(ranked_candidates), size=k, replace=False).tolist()
                    picked_dists = None
                else:
                    picked = ranked_candidates[:k]
                    picked_dists = ranked_dists[:k] if ranked_dists is not None else None

                rnas = torch.stack([self._get_rna_from_global(int(gidx)) for gidx in picked], dim=0)
                if self.control_prototype_mode == 'topk_mean' or picked_dists is None:
                    return torch.mean(rnas, dim=0)

                # topk_weighted
                weights = torch.softmax(-picked_dists / self.control_prototype_temp, dim=0).to(rnas.dtype)
                return torch.sum(weights.unsqueeze(1) * rnas, dim=0)

            def _get_rna_from_global(self, global_idx):
                row = self.full_rna[global_idx]
                if issparse(row):
                    row = row.toarray().flatten()
                return torch.tensor(row, dtype=torch.float32)

        train_doses = self.doses[train_idx] if self.doses is not None else None
        val_doses = self.doses[val_idx] if self.doses is not None else None
        test_doses = self.doses[test_idx] if self.doses is not None else None

        train_atac = self.atac_features[train_idx] if self.atac_features is not None else None
        val_atac = self.atac_features[val_idx] if self.atac_features is not None else None
        test_atac = self.atac_features[test_idx] if self.atac_features is not None else None

        train_local_batch = batch_ids[train_idx] if batch_ids is not None else None
        val_local_batch = batch_ids[val_idx] if batch_ids is not None else None
        test_local_batch = batch_ids[test_idx] if batch_ids is not None else None

        train_ds = GenerativeDataset(
            full_rna=X,
            full_atac=self.atac_features,
            sample_indices=train_idx,
            p_ids=perturb_ids[train_idx],
            c_ids=cell_line_ids[train_idx],
            p_gene_idx=perturb_gene_idx_ids[train_idx],
            p_is_control=is_control_ids[train_idx],
            p_condition_id=condition_id_ids[train_idx],
            p_source_flag=source_flag_ids[train_idx],
            doses=train_doses,
            atac_feats=train_atac,
            control_id=control_id,
            control_pool_coarse=train_control_pool_coarse,
            control_pool_fine=train_control_pool_fine if batch_ids is not None else None,
            local_batch_ids=train_local_batch,
            rna_noise=rna_noise,
            gene_mask_rate=gene_mask_rate,
            scale_rate=scale_rate,
            is_train=True,
            seed=42,
            control_match_mode=control_match_mode,
            control_match_k=control_match_k,
            control_match_scope=control_match_scope,
            control_prototype_mode=control_prototype_mode,
            control_prototype_temp=control_prototype_temp,
            knn_cache_indices=knn_cache_indices,
            knn_cache_weights=knn_cache_weights,
        )
        val_ds = GenerativeDataset(
            full_rna=X,
            full_atac=self.atac_features,
            sample_indices=val_idx,
            p_ids=perturb_ids[val_idx],
            c_ids=cell_line_ids[val_idx],
            p_gene_idx=perturb_gene_idx_ids[val_idx],
            p_is_control=is_control_ids[val_idx],
            p_condition_id=condition_id_ids[val_idx],
            p_source_flag=source_flag_ids[val_idx],
            doses=val_doses,
            atac_feats=val_atac,
            control_id=control_id,
            control_pool_coarse=ref_control_pool_coarse,
            control_pool_fine=ref_control_pool_fine,
            local_batch_ids=val_local_batch,
            is_train=False,
            seed=42,
            control_match_mode=control_match_mode,
            control_match_k=control_match_k,
            control_match_scope=control_match_scope,
            control_prototype_mode=control_prototype_mode,
            control_prototype_temp=control_prototype_temp,
            knn_cache_indices=knn_cache_indices,
            knn_cache_weights=knn_cache_weights,
        )
        test_ds = GenerativeDataset(
            full_rna=X,
            full_atac=self.atac_features,
            sample_indices=test_idx,
            p_ids=perturb_ids[test_idx],
            c_ids=cell_line_ids[test_idx],
            p_gene_idx=perturb_gene_idx_ids[test_idx],
            p_is_control=is_control_ids[test_idx],
            p_condition_id=condition_id_ids[test_idx],
            p_source_flag=source_flag_ids[test_idx],
            doses=test_doses,
            atac_feats=test_atac,
            control_id=control_id,
            control_pool_coarse=ref_control_pool_coarse,
            control_pool_fine=ref_control_pool_fine,
            local_batch_ids=test_local_batch,
            is_train=False,
            seed=42,
            control_match_mode=control_match_mode,
            control_match_k=control_match_k,
            control_match_scope=control_match_scope,
            control_prototype_mode=control_prototype_mode,
            control_prototype_temp=control_prototype_temp,
            knn_cache_indices=knn_cache_indices,
            knn_cache_weights=knn_cache_weights,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader