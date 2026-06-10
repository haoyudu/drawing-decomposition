"""
Within-Category SAE Analysis

Inputs:
  ~/Desktop/visualfeature/drawing-decomposition/sae_outputs/holdout_latents.npy
  ~/Desktop/visualfeature/drawing-decomposition/sae_outputs/holdout_metadata.parquet
  ~/Desktop/visualfeature/drawing-decomposition/sae_outputs/final_latents.npy
  ~/Desktop/visualfeature/kiddraw_clip_metadata.parquet
  ~/Desktop/visualfeature/stringent_cleaned_dataset/<filename>   (images)

Outputs (all under within_category_analysis/outputs/within_category_sae/):
  <category>_top_features.csv          top-40 features by within-category variance
  <category>_feature_heatmap.png       drawings × top-40 features activation heatmap
  <category>_top_drawings/
    feature_<NNNN>.png                 top-12 activating drawings per feature
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
base_path = os.path.expanduser("~/Desktop/visualfeature/drawing-decomposition")
data_dir  = os.path.dirname(base_path)
image_dir = os.path.join(data_dir, "stringent_cleaned_dataset")

VALIDATION_LATENTS  = os.path.join(base_path, "sae_outputs", "holdout_latents.npy")
VALIDATION_METADATA = os.path.join(base_path, "sae_outputs", "holdout_metadata.parquet")
FINAL_LATENTS       = os.path.join(base_path, "sae_outputs", "final_latents.npy")
FULL_METADATA       = os.path.join(data_dir, "kiddraw_clip_metadata.parquet")
OUTPUT_DIR          = os.path.join(base_path, "within_category_analysis", "outputs", "within_category_sae")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
HOLDOUT_CATEGORIES = ["a phone", "a cat", "a train"]
FINAL_CATEGORIES   = [
    "a scissors", "a sheep", "a key",
    "an elephant", "a snail", "a mushroom",
    "a house", "a rabbit", "a boat",
]
ALL_CATEGORIES   = HOLDOUT_CATEGORIES + FINAL_CATEGORIES

ACTIVE_THRESH    = 0.05
TOP_K_DRAWINGS   = 12
MAX_HEATMAP_ROWS = 300

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading latents and metadata...")
Z_holdout    = np.load(VALIDATION_LATENTS)
meta_holdout = pd.read_parquet(VALIDATION_METADATA)
Z_final      = np.load(FINAL_LATENTS)
meta_full    = pd.read_parquet(FULL_METADATA)

assert len(Z_holdout) == len(meta_holdout)
assert len(Z_final)   == len(meta_full)

for meta in [meta_holdout, meta_full]:
    filename_col = "filename" if "filename" in meta.columns else "image1"
    meta["clean_filename"]   = meta[filename_col].apply(lambda x: os.path.basename(str(x)))
    meta["clean_image_path"] = meta["clean_filename"].apply(lambda x: os.path.join(image_dir, x))

M = Z_final.shape[1]
print(f"  Holdout latents : {Z_holdout.shape}")
print(f"  Final latents   : {Z_final.shape}  (M={M} features)")

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_image_on_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def get_category_data(category):
    if category in HOLDOUT_CATEGORIES:
        mask = meta_holdout["category"] == category
        return Z_holdout[mask.values], meta_holdout[mask].reset_index(drop=True)
    mask = meta_full["category"] == category
    return Z_final[mask.values], meta_full[mask].reset_index(drop=True)


def get_top_features(Z_cat):
    freq    = (Z_cat > 0).mean(axis=0)
    active  = np.where(freq >= ACTIVE_THRESH)[0]
    var_all = Z_cat[:, active].var(axis=0)
    order   = np.argsort(var_all)[::-1]
    top_idx = active[order]
    return pd.DataFrame({
        "feature_idx":     top_idx,
        "variance":        var_all[order],
        "activation_freq": freq[top_idx],
        "mean_activation": Z_cat[:, top_idx].mean(axis=0),
    })


def plot_feature_heatmap(Z_cat, top_features_df, category, out_dir):
    feat_idx = top_features_df["feature_idx"].values
    Z_top    = Z_cat[:, feat_idx]
    N        = len(Z_top)

    if N > MAX_HEATMAP_ROWS:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=MAX_HEATMAP_ROWS, replace=False))
        Z_plot     = Z_top[idx]
        title_note = f" (sample {MAX_HEATMAP_ROWS}/{N})"
    else:
        Z_plot     = Z_top
        title_note = f" (all {N})"

    Z_plot = Z_plot[np.argsort(Z_plot[:, 0])[::-1]]

    n_feats = len(feat_idx)
    fig, ax = plt.subplots(figsize=(max(6, n_feats * 0.3), max(5, len(Z_plot) * 0.04)))
    sns.heatmap(Z_plot, ax=ax, cmap="YlOrRd",
                xticklabels=[f"f{i}" for i in feat_idx], yticklabels=False,
                cbar_kws={"label": "activation", "shrink": 0.4}, linewidths=0)
    ax.set_xlabel("SAE feature (active ≥5%, ordered by within-category variance)", fontsize=9)
    ax.set_ylabel("drawings", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.set_title(f"{category}  —  drawings × top features{title_note}", fontsize=10)
    plt.tight_layout()

    safe  = category.replace(" ", "_").replace("/", "_")
    path  = os.path.join(out_dir, f"{safe}_feature_heatmap.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {path}")


def plot_top_drawings(Z_cat, meta_cat, top_features_df, category, out_dir):
    safe     = category.replace(" ", "_").replace("/", "_")
    feat_dir = os.path.join(out_dir, f"{safe}_top_drawings")
    os.makedirs(feat_dir, exist_ok=True)

    for _, row in top_features_df.iterrows():
        feat        = int(row["feature_idx"])
        activations = Z_cat[:, feat]
        top_idx     = np.argsort(activations)[::-1][:TOP_K_DRAWINGS]

        n_cols = min(TOP_K_DRAWINGS, 5)
        n_rows = (TOP_K_DRAWINGS + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")

        for plot_i, data_i in enumerate(top_idx):
            ax       = axes[plot_i]
            act_val  = activations[data_i]
            meta_row = meta_cat.iloc[data_i]
            img_path = meta_row.get("clean_image_path", "")

            if os.path.exists(str(img_path)):
                try:
                    ax.imshow(load_image_on_white(img_path))
                except Exception:
                    ax.text(0.5, 0.5, "error", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            ax.set_title(f"{meta_row.get('age', '')}  act={act_val:.3f}", fontsize=7, pad=2)
            ax.axis("off")

        n_active = (activations > 0).sum()
        fig.suptitle(
            f"{category}  |  Feature {feat}  |  "
            f"active in {n_active}/{len(activations)} ({100*n_active/len(activations):.1f}%)",
            fontsize=9, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(feat_dir, f"feature_{feat:04d}.png"),
                    dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    print(f"  Grids → {feat_dir}/")


# ── Main loop ──────────────────────────────────────────────────────────────────
for category in ALL_CATEGORIES:
    safe   = category.replace(" ", "_").replace("/", "_")
    source = "validation_sae" if category in HOLDOUT_CATEGORIES else "final_sae"
    print(f"\n{'='*60}\n  {category}  [{source}]\n{'='*60}")

    Z_cat, meta_cat = get_category_data(category)
    print(f"  N drawings: {len(Z_cat)}")

    top_df   = get_top_features(Z_cat)
    n_active = int(((Z_cat > 0).mean(axis=0) >= ACTIVE_THRESH).sum())
    print(f"  Active features (≥{ACTIVE_THRESH*100:.0f}%): {n_active}  → using all {len(top_df)}")

    csv_path = os.path.join(OUTPUT_DIR, f"{safe}_top_features.csv")
    top_df.to_csv(csv_path, index=False)
    print(f"  CSV    → {csv_path}")

    plot_feature_heatmap(Z_cat, top_df, category, OUTPUT_DIR)
    plot_top_drawings(Z_cat, meta_cat, top_df, category, OUTPUT_DIR)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'Category':<22}  {'Model':<16}  {'N':>6}  {'Top feat':>10}  {'freq':>6}  {'var':>8}")
print("-" * 80)
for category in ALL_CATEGORIES:
    safe   = category.replace(" ", "_").replace("/", "_")
    source = "validation_sae" if category in HOLDOUT_CATEGORIES else "final_sae"
    df     = pd.read_csv(os.path.join(OUTPUT_DIR, f"{safe}_top_features.csv"))
    Z_cat, _ = get_category_data(category)
    r      = df.iloc[0]
    print(f"{category:<22}  {source:<16}  {len(Z_cat):>6}  "
          f"f{int(r.feature_idx):04d}  {r.activation_freq:>6.2f}  {r.variance:>8.4f}")

print("\nDone.")
