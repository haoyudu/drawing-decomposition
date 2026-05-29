"""
Visualize top-activating drawings on the held-out val split for selected SAE features.

Saves a grid PNG per feature to outputs/feature_viz/feature_{idx}.png.

Usage:
  # Visualize the 10 most-used features automatically:
  python visualize_features.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold0_sae.pt \
      --fold 0

  # Visualize specific feature indices:
  python visualize_features.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold0_sae.pt \
      --fold 0 --features 42 137 801
"""

import argparse
import os

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    raise ValueError(f"fold {fold} not found in {n_folds}-fold split")


def encode_all(model, X_tensor, batch_size, device):
    model.eval()
    Zs = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            _, latents, _ = model(batch)
            Zs.append(latents.cpu().numpy())
    return np.vstack(Zs)


def visualize_feature(feature_idx, Z_val, val_meta, top_k, out_dir):
    activations = Z_val[:, feature_idx]
    n_active = (activations > 0).sum()

    # rank by activation (descending)
    ranked = np.argsort(activations)[::-1]
    top_indices = ranked[:top_k]

    n_cols = min(top_k, 5)
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.8))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for plot_i, data_i in enumerate(top_indices):
        row = val_meta.iloc[data_i]
        img_path = row["filename"]
        act_val  = activations[data_i]

        ax = axes[plot_i]
        if os.path.exists(img_path):
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            except Exception:
                ax.text(0.5, 0.5, "load error", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)

        cat   = str(row.get("category", ""))
        age   = str(row.get("age", ""))
        title = f"{cat}\n{age}  act={act_val:.3f}"
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")

    fig.suptitle(f"Feature {feature_idx}  |  active in {n_active}/{len(activations)} val drawings",
                 fontsize=10, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"feature_{feature_idx:04d}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}  (n_active={n_active})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--metadata_path",   required=True)
    parser.add_argument("--model_path",      required=True)
    parser.add_argument("--fold",     type=int, default=0, help="Which fold's val set to use")
    parser.add_argument("--n_folds",  type=int, default=5)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--k",        type=int, default=32)
    parser.add_argument("--M",        type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--features", type=int, nargs="*", default=None,
                        help="Feature indices to visualize. If omitted, uses top --n_auto most active.")
    parser.add_argument("--n_auto",   type=int, default=10,
                        help="Number of most-active features to visualize when --features is not set")
    parser.add_argument("--top_k",    type=int, default=20,
                        help="Number of top-activating drawings to show per feature")
    parser.add_argument("--output_dir", default="outputs/feature_viz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    X_np = np.load(args.embeddings_path).astype(np.float32)
    N, D = X_np.shape
    meta = pd.read_parquet(args.metadata_path)
    assert len(meta) == N

    # Reconstruct val split (must match train_sae.py exactly)
    val_idx = get_val_indices(N, args.n_folds, args.fold, args.seed)
    val_meta = meta.iloc[val_idx].reset_index(drop=True)
    X_val = torch.from_numpy(X_np[val_idx])
    print(f"Val set: {len(val_idx)} drawings (fold {args.fold} of {args.n_folds})")

    # Load model
    model = Autoencoder(
        n_latents=args.M, n_inputs=D,
        activation=TopK(k=args.k), tied=False, normalize=True,
    ).to(device)
    sd = torch.load(args.model_path, map_location=device)
    sd_clean = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd_clean, strict=False)
    print(f"Loaded model from {args.model_path}")

    # Encode val set
    Z_val = encode_all(model, X_val, args.batch_size, device)
    print(f"Z_val shape: {Z_val.shape}")

    # Decide which features to visualize
    if args.features is not None:
        feature_list = args.features
    else:
        act_freq = (Z_val > 0).mean(axis=0)
        feature_list = np.argsort(act_freq)[::-1][:args.n_auto].tolist()
        print(f"Auto-selected top {args.n_auto} most active features: {feature_list}")

    print(f"\nVisualizing {len(feature_list)} features, top {args.top_k} drawings each...")
    for f in feature_list:
        visualize_feature(f, Z_val, val_meta, args.top_k, args.output_dir)

    print(f"\nDone. PNGs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
