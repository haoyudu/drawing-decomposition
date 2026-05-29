"""
Category × Feature heatmap for SAE activations.

For each (category, feature) pair, computes the mean activation across drawings
in that category. Selects the most category-selective features and plots a
clustered heatmap.

Usage:
  python heatmap_features.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold0_sae.pt \
      --fold 0 \
      --n_features 50 \
      --output_path outputs/category_feature_heatmap.png
"""

import argparse
import os

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from model import Autoencoder, TopK


def get_val_indices(N, n_folds, fold, seed):
    indices = np.arange(N)
    if n_folds == 1:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        split = int(0.9 * N)
        return perm[split:]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, val_idx) in enumerate(kf.split(indices)):
        if i == fold:
            return val_idx
    raise ValueError(f"fold {fold} not found")


def encode_all(model, X_tensor, batch_size, device):
    model.eval()
    Zs = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            _, latents, _ = model(batch)
            Zs.append(latents.cpu().numpy())
    return np.vstack(Zs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--metadata_path",   required=True)
    parser.add_argument("--model_path",      required=True)
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--k",          type=int, default=32)
    parser.add_argument("--M",          type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_features", type=int, default=50,
                        help="Number of selective features to show")
    parser.add_argument("--category_col", default="category")
    parser.add_argument("--freq_min",   type=float, default=0.005,
                        help="Min activation frequency to consider a feature alive")
    parser.add_argument("--output_path", default="outputs/category_feature_heatmap.png")
    parser.add_argument("--use_all_data", action="store_true",
                        help="Encode all N drawings instead of only the val fold")
    parser.add_argument("--zscore", action="store_true",
                        help="Z-score each feature column so color scale is comparable across features")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_np = np.load(args.embeddings_path).astype(np.float32)
    N, D = X_np.shape
    meta = pd.read_parquet(args.metadata_path)
    assert len(meta) == N

    if args.use_all_data:
        X_use = torch.from_numpy(X_np)
        meta_use = meta.reset_index(drop=True)
        print(f"Using all {N} drawings")
    else:
        val_idx = get_val_indices(N, args.n_folds, args.fold, args.seed)
        X_use = torch.from_numpy(X_np[val_idx])
        meta_use = meta.iloc[val_idx].reset_index(drop=True)
        print(f"Using val fold {args.fold}: {len(val_idx)} drawings")

    # Load model
    model = Autoencoder(
        n_latents=args.M, n_inputs=D,
        activation=TopK(k=args.k), tied=False, normalize=True,
    ).to(device)
    sd = torch.load(args.model_path, map_location=device)
    sd_clean = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd_clean, strict=False)
    print(f"Loaded model from {args.model_path}")

    # Encode
    Z = encode_all(model, X_use, args.batch_size, device)
    print(f"Z shape: {Z.shape}")

    # Drop dead features (never activate)
    act_freq = (Z > 0).mean(axis=0)
    alive = act_freq >= args.freq_min
    print(f"Alive features (freq >= {args.freq_min}): {alive.sum()}")

    # Build category × feature mean-activation matrix
    categories = meta_use[args.category_col].values
    cat_labels = sorted(np.unique(categories))
    n_cats = len(cat_labels)

    # Only compute over alive features to save memory
    Z_alive = Z[:, alive]           # (N_use, n_alive)
    alive_idx = np.where(alive)[0]  # original feature indices

    cat_means = np.zeros((n_cats, Z_alive.shape[1]), dtype=np.float32)
    for ci, cat in enumerate(cat_labels):
        mask = categories == cat
        cat_means[ci] = Z_alive[mask].mean(axis=0)

    # Select most category-selective features:
    # selectivity = std across categories / (mean across categories + eps)
    # High value = feature is much stronger for some categories than others
    selectivity = cat_means.std(axis=0) / (cat_means.mean(axis=0) + 1e-8)
    top_local = np.argsort(selectivity)[::-1][:args.n_features]
    top_global = alive_idx[top_local]  # original feature indices

    heatmap_data = cat_means[:, top_local]  # (n_cats, n_features)
    col_labels = [f"f{i}" for i in top_global]

    if args.zscore:
        col_mean = heatmap_data.mean(axis=0, keepdims=True)
        col_std  = heatmap_data.std(axis=0, keepdims=True) + 1e-8
        plot_data = (heatmap_data - col_mean) / col_std
    else:
        plot_data = heatmap_data

    df = pd.DataFrame(plot_data, index=cat_labels, columns=col_labels)

    # Clustered heatmap
    figsize = (max(12, args.n_features * 0.25), max(8, n_cats * 0.25))
    g = sns.clustermap(
        df,
        figsize=figsize,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.ax_heatmap.set_xlabel("SAE feature (by index)", fontsize=10)
    g.ax_heatmap.set_ylabel("Drawing category", fontsize=10)
    g.ax_heatmap.tick_params(axis="x", labelsize=6, rotation=90)
    g.ax_heatmap.tick_params(axis="y", labelsize=7)
    zscore_note = " (z-scored)" if args.zscore else ""
    g.figure.suptitle(
        f"Category × Feature mean activation{zscore_note}\n"
        f"Top {args.n_features} most category-selective features  |  {n_cats} categories",
        y=1.01, fontsize=11,
    )

    g.savefig(args.output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {args.output_path}")
    print(f"Top selective feature indices: {top_global[:10].tolist()} ...")


if __name__ == "__main__":
    main()
