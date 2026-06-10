"""
Per-category PCA scree analysis.

For each of the 12 selected categories, runs PCA on the CLIP embeddings of
drawings in that category and finds how many components explain 80% and 90% of variance.

Outputs:
  outputs/pca/pca_scree_all_categories.png  — 4x3 grid of scree curves (80% + 90% marked)
  outputs/pca/pca_variance_sweep.png        — bar chart comparing 80% vs 90% across categories
  outputs/pca/pca_variance_summary.csv      — n_components at 80% and 90% per category
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EMBEDDINGS_PATH = "../../kiddraw_clip_embeddings.npy"
METADATA_PATH   = "../../kiddraw_clip_metadata.parquet"
OUTPUT_DIR      = "outputs/pca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = [
    # > 1000 drawings
    "a phone", "a cat", "a train", "a scissors", "a sheep", "a key",
    # ~ 600 drawings
    "an elephant", "a snail", "a mushroom", "a house", "a rabbit", "a boat",
]

HOLDOUT_CATEGORIES = {"a phone", "a cat", "a train"}


def pca_scree(X):
    """Fit full PCA on X; return (cumulative_variance, n_components_at_80pct, n_components_at_90pct)."""
    pca = PCA().fit(X)
    cev = np.cumsum(pca.explained_variance_ratio_)
    n_80 = int(np.searchsorted(cev, 0.80)) + 1
    n_90 = int(np.searchsorted(cev, 0.90)) + 1
    return cev, n_80, n_90


def main():
    print("Loading embeddings and metadata...")
    X_all = np.load(EMBEDDINGS_PATH).astype(np.float32)
    meta  = pd.read_parquet(METADATA_PATH)
    assert len(X_all) == len(meta)
    print(f"  Embeddings: {X_all.shape}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    summary_rows = []

    for i, category in enumerate(CATEGORIES):
        mask  = meta["category"] == category
        X_cat = X_all[mask.values]
        n     = len(X_cat)
        print(f"\n{category}  (n={n})")

        cev, n_80, n_90 = pca_scree(X_cat)
        summary_rows.append({"category": category, "n_drawings": n,
                              "n_components_80pct": n_80, "n_components_90pct": n_90,
                              "holdout": category in HOLDOUT_CATEGORIES})
        print(f"  80% variance: {n_80} components  |  90% variance: {n_90} components")

        ax = axes[i]
        x  = np.arange(1, len(cev) + 1)
        ax.plot(x, cev, lw=1.5, color="steelblue")
        ax.axvline(n_80, color="darkorange", linestyle="--", lw=1.2,
                   label=f"80% → {n_80}")
        ax.axvline(n_90, color="crimson", linestyle="--", lw=1.2,
                   label=f"90% → {n_90}")
        ax.axhline(0.80, color="darkorange", linestyle=":", lw=0.6)
        ax.axhline(0.90, color="crimson", linestyle=":", lw=0.6)
        ax.set_xlim(0, min(len(cev), 150))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("components", fontsize=8)
        ax.set_ylabel("cumul. var.", fontsize=8)
        ax.tick_params(labelsize=7)

        marker = " *" if category in HOLDOUT_CATEGORIES else ""
        ax.set_title(f"{category}{marker}  (n={n})", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Per-category PCA scree — CLIP embeddings\n"
        "* = held-out category (validation SAE)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "pca_scree_all_categories.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "pca_variance_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved: {csv_path}")
    print()
    print(summary_df.to_string(index=False))

    # Variance sweep bar chart: 80% vs 90% side by side per category
    labels    = [c.replace("a ", "").replace("an ", "") for c in CATEGORIES]
    x         = np.arange(len(CATEGORIES))
    width     = 0.38
    vals_80   = summary_df["n_components_80pct"].values
    vals_90   = summary_df["n_components_90pct"].values
    holdouts  = summary_df["holdout"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_80 = ax.bar(x - width / 2, vals_80, width, label="80% variance",
                     color="darkorange", alpha=0.85)
    bars_90 = ax.bar(x + width / 2, vals_90, width, label="90% variance",
                     color="steelblue", alpha=0.85)

    # Hatch holdout categories
    for idx, is_holdout in enumerate(holdouts):
        if is_holdout:
            bars_80[idx].set_hatch("//")
            bars_90[idx].set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("n components", fontsize=10)
    ax.set_title(
        "PCA components needed for 80% vs 90% variance — per category\n"
        "(hatched = held-out categories)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(vals_90) * 1.15)

    for bar in list(bars_80) + list(bars_90):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    sweep_path = os.path.join(OUTPUT_DIR, "pca_variance_sweep.png")
    plt.savefig(sweep_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Variance sweep saved: {sweep_path}")


if __name__ == "__main__":
    main()
