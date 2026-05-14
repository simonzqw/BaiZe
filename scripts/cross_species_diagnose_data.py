import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

CONTROL_ALIASES = {"CTRL", "CTRL1", "ctrl", "Control", "vehicle", "Vehicle"}


def dense_mean(x):
    if x.shape[0] == 0:
        raise ValueError("Cannot compute mean for empty AnnData slice.")
    return np.asarray(x.mean(axis=0)).ravel() if issparse(x) else np.asarray(x).mean(axis=0)


def normalize_perturbation(arr, control_key="control"):
    arr = np.asarray(arr, dtype=object)
    arr = np.array([str(x) for x in arr], dtype=object)
    mask = np.isin(arr, list(CONTROL_ALIASES | {control_key}))
    arr[mask] = str(control_key)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_h5ad", required=True)
    p.add_argument("--mouse_h5ad", required=True)
    p.add_argument("--perturb_col", default="perturbation")
    p.add_argument("--context_col", default="cell_context")
    p.add_argument("--species_col", default="species")
    p.add_argument("--split_col", default="split")
    p.add_argument("--control_key", default="control")
    p.add_argument("--atac_key", default="atac_feat")
    p.add_argument("--perturbations", nargs="+", default=["ARID1A", "PDCD1"])
    args = p.parse_args()

    adata = sc.read_h5ad(args.combined_h5ad)
    mouse = sc.read_h5ad(args.mouse_h5ad)

    print("gene_order_equal:", np.array_equal(adata.var_names.values, mouse.var_names.values))
    print("combined shape:", adata.shape, "mouse shape:", mouse.shape)
    print("species counts\n", adata.obs[args.species_col].value_counts())
    print("split counts\n", adata.obs[args.split_col].value_counts())

    mp = normalize_perturbation(mouse.obs[args.perturb_col].values, args.control_key)
    print("mouse perturb counts after normalization\n", pd.Series(mp).value_counts())
    print("mouse context counts\n", mouse.obs[args.context_col].value_counts())

    if args.atac_key in adata.obsm:
        print("combined atac", adata.obsm[args.atac_key].shape)
    else:
        print("combined missing atac:", args.atac_key)
    if args.atac_key in mouse.obsm:
        print("mouse atac", mouse.obsm[args.atac_key].shape)
    else:
        print("mouse missing atac:", args.atac_key)

    human = adata[adata.obs[args.species_col].astype(str).values == "human"].copy()
    hp = normalize_perturbation(human.obs[args.perturb_col].values, args.control_key)

    print("\n=== human context x perturbation ===")
    print(pd.crosstab(human.obs[args.context_col].astype(str).values, hp))
    print("\n=== mouse context x perturbation ===")
    print(pd.crosstab(mouse.obs[args.context_col].astype(str).values, mp))

    h_ctrl = human[hp == args.control_key]
    m_ctrl = mouse[mp == args.control_key]
    print("\nhuman control cells:", h_ctrl.n_obs)
    print("mouse control cells:", m_ctrl.n_obs)
    if h_ctrl.n_obs == 0:
        raise ValueError("No human control cells found after normalization.")
    if m_ctrl.n_obs == 0:
        raise ValueError("No mouse control cells found after normalization.")

    h_ctrl_mean = dense_mean(h_ctrl.X)
    m_ctrl_mean = dense_mean(m_ctrl.X)

    print("\n=== human vs mouse delta norm / cosine ===")
    for g in args.perturbations:
        h_g = human[hp == g]
        m_g = mouse[mp == g]
        if h_g.n_obs == 0 or m_g.n_obs == 0:
            print(g, "missing in human or mouse", "human_n", h_g.n_obs, "mouse_n", m_g.n_obs)
            continue
        h_delta = dense_mean(h_g.X) - h_ctrl_mean
        m_delta = dense_mean(m_g.X) - m_ctrl_mean
        h_norm = np.linalg.norm(h_delta)
        m_norm = np.linalg.norm(m_delta)
        cos = float(np.dot(h_delta, m_delta) / (h_norm * m_norm + 1e-8))
        print(g, "human_n", h_g.n_obs, "mouse_n", m_g.n_obs, "human_norm", float(h_norm), "mouse_norm", float(m_norm), "cos", cos)


if __name__ == "__main__":
    main()
