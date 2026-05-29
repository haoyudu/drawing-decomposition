"""
Category × Feature heatmap for SAE activations.

For each (category, feature) pair, computes the mean activation across drawings
in that category. Selects the most category-selective features and plots a heatmap
with rows/columns ordered so high values fall along the diagonal.

Usage:
  python heatmap_features.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold0_sae.pt \
      --fold 0 \
      --n_features 50 \
      --zscore \
      --output_path outputs/category_feature_heatmap.png

  # Also save top drawings per category-feature pair:
  python heatmap_features.py ... --save_top_images --top_k 10
"""

import argparse
import os

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def diagonal_order(mat):
    """
    Reorder rows and columns of mat so high values fall near the diagonal.

    Strategy:
      1. Sort columns by which row has the highest value (argmax over rows).
      2. After sorting columns, sort rows by which column position their peak falls at.
    Returns (row_order, col_order) as index arrays.
    """
    col_peak_row = mat.argmax(axis=0)          # for each col, which row peaks it
    col_order = np.argsort(col_peak_row)       # sort cols by their peak row

    mat_cols_sorted = mat[:, col_order]
    row_peak_col = mat_cols_sorted.argmax(axis=1)  # for each row, which col position it peaks at
    row_order = np.argsort(row_peak_col)

    return row_order, col_order


def save_collage(
    feature_global_indices, primary_cat_per_feature, cat_labels,
    Z, meta_use, categories, top_k, out_dir,
):
    """
    Save one collage PNG per feature: a 1-row grid of top_k drawings from
    the feature's primary category. Files saved to out_dir/.
    """
    os.makedirs(out_dir, exist_ok=True)
    cell = 1.8  # inches per image cell

    for feat_global, primary_cat_idx in zip(feature_global_indices, primary_cat_per_feature):
        cat_name = cat_labels[primary_cat_idx]
        cat_mask = categories == cat_name
        cat_indices = np.where(cat_mask)[0]

        fig, axes = plt.subplots(1, top_k, figsize=(top_k * cell, cell * 1.4))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")

        if len(cat_indices) > 0:
            activations = Z[cat_indices, feat_global]
            ranked = np.argsort(activations)[::-1][:top_k]
            top_data_indices = cat_indices[ranked]

            for ki, data_i in enumerate(top_data_indices):
                row = meta_use.iloc[data_i]
                img_path = row["filename"]
                act_val = activations[ranked[ki]]
                ax = axes[ki]
                if os.path.exists(img_path):
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    except Exception:
                        ax.text(0.5, 0.5, "err", ha="center", va="center",
                                transform=ax.transAxes, fontsize=7)
                else:
                    ax.text(0.5, 0.5, "?", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7)
                age = str(row.get("age", ""))
                ax.set_title(f"{age}  {act_val:.2f}", fontsize=7, pad=2)

        n_active = (Z[cat_indices, feat_global] > 0).sum() if len(cat_indices) > 0 else 0
        fig.suptitle(
            f"Feature {feat_global}  |  {cat_name}  |  "
            f"active in {n_active}/{len(cat_indices)} drawings",
            fontsize=9,
        )
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"feature_{feat_global:04d}_{cat_name}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


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
                        help="Number of most category-selective features to show")
    parser.add_argument("--category_col", default="category")
    parser.add_argument("--freq_min",   type=float, default=0.005,
                        help="Min activation frequency to consider a feature alive")
    parser.add_argument("--output_path", default="outputs/category_feature_heatmap.png")
    parser.add_argument("--use_all_data", action="store_true",
                        help="Encode all N drawings instead of only the val fold")
    parser.add_argument("--zscore", action="store_true",
                        help="Z-score each feature column so scale is comparable across features")
    parser.add_argument("--save_top_images", action="store_true",
                        help="Save a PNG per feature showing top drawings from its primary category")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top drawings per feature (used with --save_top_images)")
    parser.add_argument("--top_images_dir", default=None,
                        help="Output dir for top-image PNGs (default: <output_path stem>_top_images/)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
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

    # Drop dead features
    act_freq = (Z > 0).mean(axis=0)
    alive = act_freq >= args.freq_min
    print(f"Alive features (freq >= {args.freq_min}): {alive.sum()}")

    # Build category × feature mean-activation matrix (over alive features only)
    categories = meta_use[args.category_col].values
    cat_labels = sorted(np.unique(categories))
    n_cats = len(cat_labels)

    Z_alive = Z[:, alive]
    alive_idx = np.where(alive)[0]

    cat_means = np.zeros((n_cats, Z_alive.shape[1]), dtype=np.float32)
    for ci, cat in enumerate(cat_labels):
        mask = categories == cat
        cat_means[ci] = Z_alive[mask].mean(axis=0)

    # Select most category-selective features
    # selectivity = std across categories / (mean + eps): high = fires much more for some cats
    selectivity = cat_means.std(axis=0) / (cat_means.mean(axis=0) + 1e-8)
    top_local = np.argsort(selectivity)[::-1][:args.n_features]
    top_global = alive_idx[top_local]

    heatmap_data = cat_means[:, top_local]   # (n_cats, n_features)

    # Diagonal ordering (on raw activations, before optional z-score)
    row_order, col_order = diagonal_order(heatmap_data)
    heatmap_data = heatmap_data[row_order, :][:, col_order]
    cat_labels_ordered = [cat_labels[i] for i in row_order]
    top_global_ordered = top_global[col_order]
    col_labels = [f"f{i}" for i in top_global_ordered]

    # Primary category per feature (after reordering, for save_top_images)
    primary_cat_per_feature = row_order[heatmap_data.argmax(axis=0)]

    # Optional z-score (after reordering so it doesn't affect ordering)
    if args.zscore:
        col_mean = heatmap_data.mean(axis=0, keepdims=True)
        col_std  = heatmap_data.std(axis=0, keepdims=True) + 1e-8
        plot_data = (heatmap_data - col_mean) / col_std
    else:
        plot_data = heatmap_data

    df = pd.DataFrame(plot_data, index=cat_labels_ordered, columns=col_labels)

    # Heatmap (no dendrogram)
    cell_w, cell_h = 0.28, 0.28
    figsize = (max(10, args.n_features * cell_w + 2), max(6, n_cats * cell_h + 2))
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df, ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.5, "label": "mean activation (z-scored)" if args.zscore else "mean activation"},
        linewidths=0,
    )
    ax.set_xlabel("SAE feature (by index)", fontsize=10)
    ax.set_ylabel("Drawing category", fontsize=10)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    zscore_note = " (z-scored)" if args.zscore else ""
    ax.set_title(
        f"Category × Feature mean activation{zscore_note}\n"
        f"Top {args.n_features} most category-selective features  |  {n_cats} categories",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {args.output_path}")
    print(f"Top selective feature indices (ordered): {top_global_ordered[:10].tolist()} ...")

    # Optionally save per-feature collages
    if args.save_top_images:
        collage_dir = args.top_images_dir or (
            os.path.splitext(args.output_path)[0] + "_collages"
        )
        print(f"\nSaving top-{args.top_k} collages per feature to {collage_dir}/")
        save_collage(
            top_global_ordered, primary_cat_per_feature, cat_labels,
            Z, meta_use, categories, args.top_k, collage_dir,
        )


if __name__ == "__main__":
    main()
