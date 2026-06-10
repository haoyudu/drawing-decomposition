# -*- coding: utf-8 -*-
"""
Run PCA on Kiddraw CLIP embeddings (full dataset, no train/test split).

PCA path:
  1. Load CLIP embeddings.
  2. Fit PCA(n_components=55) directly on embeddings.
  3. Save PCA scores, top/bottom images, category heatmaps, normalized heatmap,
     z-scored heatmap, and age trajectories.

Author: L. Kemmelmeier
"""

#%% imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error


#%% paths and parameters

base_path = os.path.expanduser("~/Documents/GitHub/drawing-decomposition")

data_dir = os.path.join(base_path, "data")
image_dir = os.path.join(data_dir, "stringent_cleaned_dataset")
out_dir = os.path.join(base_path, "pca_outputs")

os.makedirs(out_dir, exist_ok=True)

embeddings_file = os.path.join(data_dir, "kiddraw_clip_embeddings.npy")
metadata_file = os.path.join(data_dir, "kiddraw_clip_metadata.parquet")

n_components = 55
random_state = 0
n_extreme_images = 20


#%% load data

X = np.load(embeddings_file)
metadata = pd.read_parquet(metadata_file).copy()

metadata["clean_filename"] = metadata["image1"].apply(
    lambda x: os.path.basename(str(x))
)

metadata["clean_image_path"] = metadata["clean_filename"].apply(
    lambda x: os.path.join(image_dir, x)
)

print(f"Loaded {X.shape[0]} drawings with {X.shape[1]} CLIP features")


#%% fit PCA

pca = PCA(
    n_components=n_components,
    random_state=random_state,
)

S = pca.fit_transform(X)

print(
    f"PCA explained variance with {n_components} components: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)


#%% save PCA outputs

np.save(os.path.join(out_dir, "S.npy"), S)
np.save(os.path.join(out_dir, "pca_components.npy"), pca.components_)
np.save(os.path.join(out_dir, "pca_explained_variance_ratio.npy"), pca.explained_variance_ratio_)

metadata.to_parquet(os.path.join(out_dir, "metadata.parquet"))


#%% attach PCA scores to metadata

pca_col_names = [f"pca_{i:03d}" for i in range(n_components)]

S_df = pd.DataFrame(S, columns=pca_col_names)

metadata_pca = pd.concat(
    [metadata.reset_index(drop=True), S_df],
    axis=1,
)

metadata_pca.to_parquet(os.path.join(out_dir, "metadata_pca.parquet"))


#%% save top and bottom image names for each PCA component

component_extreme_rows = []

for component in range(n_components):

    component_col = f"pca_{component:03d}"

    low_df = (
        metadata_pca
        .sort_values(component_col)
        .head(n_extreme_images)
    )

    high_df = (
        metadata_pca
        .sort_values(component_col, ascending=False)
        .head(n_extreme_images)
    )

    for direction, df in [("low", low_df), ("high", high_df)]:

        for rank, (_, row) in enumerate(df.iterrows(), start=1):

            component_extreme_rows.append({
                "component": component_col,
                "direction": direction,
                "rank": rank,
                "score": row[component_col],
                "filename": row["clean_filename"],
                "image_path": row["clean_image_path"],
                "category": row["category"],
                "age_numeric": row["age_numeric"],
            })

component_extremes_df = pd.DataFrame(component_extreme_rows)

component_extremes_df.to_csv(
    os.path.join(out_dir, "pca_component_top_bottom_filenames.csv"),
    index=False,
)


#%% plot top and bottom images for each PCA component

component_plot_dir = os.path.join(out_dir, "component_image_grids")
os.makedirs(component_plot_dir, exist_ok=True)


def load_image_on_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img)
    return img.convert("RGB")


def plot_component_extremes(metadata_pca, component, n_images=12):

    component_col = f"pca_{component:03d}"

    low_df = (
        metadata_pca
        .sort_values(component_col)
        .head(n_images)
    )

    high_df = (
        metadata_pca
        .sort_values(component_col, ascending=False)
        .head(n_images)
    )

    fig, axes = plt.subplots(
        2,
        n_images,
        figsize=(n_images * 1.7, 6.0),
    )

    plt.subplots_adjust(
        hspace=0.55,
        wspace=0.15,
    )

    for i, (_, row) in enumerate(low_df.iterrows()):

        img = load_image_on_white(row["clean_image_path"])

        axes[0, i].imshow(img)
        axes[0, i].axis("off")

        axes[0, i].set_title(
            f"{row[component_col]:.2f}",
            fontsize=8,
        )

        axes[0, i].text(
            0.5,
            -0.04,
            str(row["category"]),
            ha="center",
            va="top",
            transform=axes[0, i].transAxes,
            fontsize=8,
        )

    for i, (_, row) in enumerate(high_df.iterrows()):

        img = load_image_on_white(row["clean_image_path"])

        axes[1, i].imshow(img)
        axes[1, i].axis("off")

        axes[1, i].set_title(
            f"{row[component_col]:.2f}",
            fontsize=8,
        )

        axes[1, i].text(
            0.5,
            -0.04,
            str(row["category"]),
            ha="center",
            va="top",
            transform=axes[1, i].transAxes,
            fontsize=8,
        )

    axes[0, 0].set_ylabel("Low", fontsize=12)
    axes[1, 0].set_ylabel("High", fontsize=12)

    fig.suptitle(
        f"PCA Component {component}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    save_path = os.path.join(
        component_plot_dir,
        f"pca_component_{component:03d}_extremes.png",
    )

    plt.savefig(save_path, dpi=300, facecolor="white")
    plt.close()


for component in range(n_components):
    plot_component_extremes(
        metadata_pca=metadata_pca,
        component=component,
        n_images=12,
    )


#%% category-component means

category_component_means = (
    metadata_pca
    .groupby("category")[pca_col_names]
    .mean()
)

category_component_means.to_csv(
    os.path.join(out_dir, "category_component_mean_scores.csv"),
)


#%% helper function for ordering heatmaps

def order_category_component_matrix(df, component_prefix="pca_", use_abs=True):

    ranking_values = df.abs() if use_abs else df.copy()

    strongest_component = ranking_values.idxmax(axis=1)

    strongest_component_num = (
        strongest_component
        .str.replace(component_prefix, "", regex=False)
        .astype(int)
    )

    row_order = strongest_component_num.sort_values().index

    ordered_df = df.loc[row_order]
    ranking_values = ranking_values.loc[row_order]

    component_best_category = ranking_values.idxmax(axis=0)

    category_rank = {
        cat: i
        for i, cat in enumerate(row_order)
    }

    component_order = (
        pd.Series(component_best_category, index=ranking_values.columns)
        .map(category_rank)
        .sort_values()
        .index
    )

    ordered_df = ordered_df[component_order]

    component_labels = [
        str(int(col.replace(component_prefix, "")))
        for col in ordered_df.columns
    ]

    return ordered_df, component_labels


#%% category-component heatmap

category_heatmap_dir = os.path.join(
    out_dir,
    "category_component_heatmaps",
)

os.makedirs(category_heatmap_dir, exist_ok=True)

heatmap_df, component_labels = order_category_component_matrix(
    category_component_means,
    component_prefix="pca_",
    use_abs=True,
)

fig, ax = plt.subplots(figsize=(22, 18))

im = ax.imshow(
    heatmap_df.values,
    aspect="auto",
    interpolation="nearest",
)

ax.set_yticks(np.arange(len(heatmap_df.index)))
ax.set_yticklabels(
    heatmap_df.index,
    fontsize=18,
    fontweight="bold",
)

ax.set_xticks(np.arange(len(heatmap_df.columns)))
ax.set_xticklabels(
    component_labels,
    rotation=90,
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel(
    "PCA Component",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.set_ylabel(
    "Category",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.tick_params(axis="x", length=4, labelsize=13)
ax.tick_params(axis="y", length=0, labelsize=18)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.02,
)

cbar.set_label(
    "Mean PCA Score",
    fontsize=20,
    fontweight="bold",
    labelpad=15,
)

cbar.ax.tick_params(labelsize=16)

plt.tight_layout()

plt.savefig(
    os.path.join(
        category_heatmap_dir,
        "category_component_heatmap.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


#%% normalized category-component heatmap (within PCA component)

category_component_means_norm = (
    category_component_means
    - category_component_means.min(axis=0)
)

category_component_means_norm = (
    category_component_means_norm
    / category_component_means_norm.max(axis=0)
)

heatmap_df_norm, component_labels_norm = order_category_component_matrix(
    category_component_means_norm,
    component_prefix="pca_",
    use_abs=False,
)

fig, ax = plt.subplots(figsize=(22, 18))

im = ax.imshow(
    heatmap_df_norm.values,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)

ax.set_yticks(np.arange(len(heatmap_df_norm.index)))
ax.set_yticklabels(
    heatmap_df_norm.index,
    fontsize=18,
    fontweight="bold",
)

ax.set_xticks(np.arange(len(heatmap_df_norm.columns)))
ax.set_xticklabels(
    component_labels_norm,
    rotation=90,
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel(
    "PCA Component",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.set_ylabel(
    "Category",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.tick_params(axis="x", length=4, labelsize=13)
ax.tick_params(axis="y", length=0, labelsize=18)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.02,
)

cbar.set_label(
    "Normalized Mean PCA Score",
    fontsize=20,
    fontweight="bold",
    labelpad=15,
)

cbar.ax.tick_params(labelsize=16)

plt.tight_layout()

plt.savefig(
    os.path.join(
        category_heatmap_dir,
        "category_component_heatmap_normalized.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


#%% z-scored category-component heatmap (within PCA component)

category_component_means_z = (
    category_component_means
    - category_component_means.mean(axis=0)
) / category_component_means.std(axis=0)

heatmap_df_z, component_labels_z = order_category_component_matrix(
    category_component_means_z,
    component_prefix="pca_",
    use_abs=False,
)

fig, ax = plt.subplots(figsize=(22, 18))

vmax = np.nanmax(np.abs(heatmap_df_z.values))

im = ax.imshow(
    heatmap_df_z.values,
    aspect="auto",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
)

ax.set_yticks(np.arange(len(heatmap_df_z.index)))
ax.set_yticklabels(
    heatmap_df_z.index,
    fontsize=18,
    fontweight="bold",
)

ax.set_xticks(np.arange(len(heatmap_df_z.columns)))
ax.set_xticklabels(
    component_labels_z,
    rotation=90,
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel(
    "PCA Component",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.set_ylabel(
    "Category",
    fontsize=24,
    fontweight="bold",
    labelpad=15,
)

ax.tick_params(axis="x", length=4, labelsize=13)
ax.tick_params(axis="y", length=0, labelsize=18)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.02,
)

cbar.set_label(
    "Z-scored Mean PCA Score",
    fontsize=20,
    fontweight="bold",
    labelpad=15,
)

cbar.ax.tick_params(labelsize=16)

plt.tight_layout()

plt.savefig(
    os.path.join(
        category_heatmap_dir,
        "category_component_heatmap_zscore.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


#%% age prediction from PCA components (full sample; metrics are in-sample)

age_model = RidgeCV(
    alphas=np.logspace(-3, 3, 20),
)

X_age = metadata_pca[pca_col_names].values
y_age = metadata_pca["age_numeric"].values

age_model.fit(X_age, y_age)

age_pred = age_model.predict(X_age)

age_r2 = r2_score(y_age, age_pred)
age_mae = mean_absolute_error(y_age, age_pred)

age_results = pd.DataFrame({
    "metric": ["r2", "mean_absolute_error", "alpha"],
    "value": [age_r2, age_mae, age_model.alpha_],
})

age_results.to_csv(
    os.path.join(out_dir, "pca_age_prediction_summary.csv"),
    index=False,
)

age_weights = pd.DataFrame({
    "component": pca_col_names,
    "weight": age_model.coef_,
    "abs_weight": np.abs(age_model.coef_),
}).sort_values("abs_weight", ascending=False)

age_weights.to_csv(
    os.path.join(out_dir, "pca_age_prediction_weights.csv"),
    index=False,
)


#%% age trajectories for top age-related PCA components

age_plot_dir = os.path.join(out_dir, "age_trajectories")
os.makedirs(age_plot_dir, exist_ok=True)

top_age_components = age_weights["component"].head(10).tolist()

for comp in top_age_components:

    age_summary = (
        metadata_pca
        .groupby("age_numeric")[comp]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    plt.errorbar(
        age_summary["age_numeric"],
        age_summary["mean"],
        yerr=age_summary["sem"],
        marker="o",
        linewidth=2,
        capsize=3,
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Age")
    plt.ylabel("Mean PCA Score")
    plt.title(f"Age trajectory: {comp}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            age_plot_dir,
            f"{comp}_age_trajectory.png",
        ),
        dpi=300,
    )

    plt.close()


#%% done

print(f"Done. Outputs saved to: {out_dir}")