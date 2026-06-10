# -*- coding: utf-8 -*-
"""
Run ICA on Kiddraw CLIP embeddings (full dataset, no train/test split).

ICA path matches geometry_analysis.build_representations (ICA branch), in order:

  1. StandardScaler on CLIP embeddings (per-feature z-score).
  2. PCA(n_components) on the scaled matrix — dimensionality reduction only;
     sklearn PCA defaults to whiten=False (no sphering / unit-variance PCA space).
  3. FastICA on the PCA scores with whiten=False — no additional FastICA internal
     whitening (avoids FastICA's expensive full-dataset SVD on raw CLIP dims).

Author: L. Kemmelmeier
"""

#%% imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error


#%% paths and parameters

base_path = os.path.expanduser("~/Documents/GitHub/drawing-decomposition")

data_dir = os.path.join(base_path, "data")
image_dir = os.path.join(data_dir, "stringent_cleaned_dataset")
out_dir = os.path.join(base_path, "ica_outputs")

os.makedirs(out_dir, exist_ok=True)

embeddings_file = os.path.join(data_dir, "kiddraw_clip_embeddings.npy")
metadata_file = os.path.join(data_dir, "kiddraw_clip_metadata.parquet")

n_components = 55
random_state = 0
# FastICA max_iter (geometry_analysis.build_representations uses 200); increase if
# sklearn emits a ConvergenceWarning on the full corpus.
ica_max_iter = 200
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


#%% ICA path — same steps as geometry_analysis.build_representations

# Step 1: standardize CLIP features (ICA is unstable on arbitrary feature scales).
scaler_ica = StandardScaler()
X_ica = scaler_ica.fit_transform(X)

# Step 2: PCA on standardized embeddings only (not raw CLIP). Reduces to
# n_components before FastICA; PCA(..., whiten=False) is sklearn default.
pca_ica = PCA(n_components=n_components, random_state=random_state)
X_ica_pca = pca_ica.fit_transform(X_ica)

# Step 3: FastICA on PCA scores; whiten=False — no unit-variance whitening inside
# FastICA (geometry_analysis uses this so FastICA does not re-whiten the subspace).
ica = FastICA(
    n_components=n_components,
    random_state=random_state,
    max_iter=ica_max_iter,
    whiten=False,
)

S = ica.fit_transform(X_ica_pca)


#%% save ICA outputs

np.save(os.path.join(out_dir, "S.npy"), S)

np.save(os.path.join(out_dir, "ica_components.npy"), ica.components_)
np.save(os.path.join(out_dir, "ica_pca_components.npy"), pca_ica.components_)

metadata.to_parquet(os.path.join(out_dir, "metadata.parquet"))


#%% attach ICA scores to metadata

ica_col_names = [f"ica_{i:03d}" for i in range(n_components)]

S_df = pd.DataFrame(S, columns=ica_col_names)

metadata_ica = pd.concat(
    [metadata.reset_index(drop=True), S_df],
    axis=1,
)

metadata_ica.to_parquet(os.path.join(out_dir, "metadata_ica.parquet"))


#%% save top and bottom image names for each ICA component

component_extreme_rows = []

for component in range(n_components):

    component_col = f"ica_{component:03d}"

    low_df = (
        metadata_ica
        .sort_values(component_col)
        .head(n_extreme_images)
    )

    high_df = (
        metadata_ica
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
    os.path.join(out_dir, "ica_component_top_bottom_filenames.csv"),
    index=False,
)


#%% plot top and bottom images for each ICA component

component_plot_dir = os.path.join(out_dir, "component_image_grids")
os.makedirs(component_plot_dir, exist_ok=True)


def load_image_on_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img)
    return img.convert("RGB")


def plot_component_extremes(metadata_ica, component, n_images=12):

    component_col = f"ica_{component:03d}"

    low_df = (
        metadata_ica
        .sort_values(component_col)
        .head(n_images)
    )

    high_df = (
        metadata_ica
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
        f"ICA Component {component}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    save_path = os.path.join(
        component_plot_dir,
        f"ica_component_{component:03d}_extremes.png",
    )

    plt.savefig(save_path, dpi=300, facecolor="white")
    plt.close()


for component in range(n_components):
    plot_component_extremes(
        metadata_ica=metadata_ica,
        component=component,
        n_images=12,
    )


#%% category-component means

category_component_means = (
    metadata_ica
    .groupby("category")[ica_col_names]
    .mean()
)

category_component_means.to_csv(
    os.path.join(out_dir, "category_component_mean_scores.csv"),
)


#%% helper function for ordering heatmaps

def order_category_component_matrix(df, component_prefix="ica_", use_abs=True):

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
    component_prefix="ica_",
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
    "ICA Component",
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
    "Mean ICA Score",
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


#%% ranked heatmap variant (same data, abs-ordered)

heatmap_df_ranked, component_labels_ranked = order_category_component_matrix(
    category_component_means,
    component_prefix="ica_",
    use_abs=True,
)

fig, ax = plt.subplots(figsize=(22, 18))

im = ax.imshow(
    heatmap_df_ranked.values,
    aspect="auto",
    interpolation="nearest",
)

ax.set_yticks(np.arange(len(heatmap_df_ranked.index)))
ax.set_yticklabels(
    heatmap_df_ranked.index,
    fontsize=18,
    fontweight="bold",
)

ax.set_xticks(np.arange(len(heatmap_df_ranked.columns)))
ax.set_xticklabels(
    component_labels_ranked,
    rotation=90,
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel(
    "ICA Component",
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

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label(
    "Mean ICA Score",
    fontsize=20,
    fontweight="bold",
    labelpad=15,
)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()

plt.savefig(
    os.path.join(
        category_heatmap_dir,
        "category_component_heatmap_ranked.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


#%% normalized category-component heatmap (within ICA component)

# normalize each ICA component across categories
# brighter values = categories that rely more heavily on that component
# relative to other categories

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
    component_prefix="ica_",
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
    "ICA Component",
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
    "Normalized Mean ICA Score",
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

#%% z-scored category-component heatmap (within ICA component)

# z-score each ICA component across categories
# positive values = category scores higher than average for that component
# negative values = category scores lower than average for that component

category_component_means_z = (
    category_component_means
    - category_component_means.mean(axis=0)
) / category_component_means.std(axis=0)

heatmap_df_z, component_labels_z = order_category_component_matrix(
    category_component_means_z,
    component_prefix="ica_",
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
    "ICA Component",
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
    "Z-scored Mean ICA Score",
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


#%% age prediction from ICA components (full sample; metrics are in-sample)

age_model = RidgeCV(
    alphas=np.logspace(-3, 3, 20),
)

X_age = metadata_ica[ica_col_names].values
y_age = metadata_ica["age_numeric"].values

age_model.fit(X_age, y_age)

age_pred = age_model.predict(X_age)

age_r2 = r2_score(y_age, age_pred)
age_mae = mean_absolute_error(y_age, age_pred)

age_results = pd.DataFrame({
    "metric": ["r2", "mean_absolute_error", "alpha"],
    "value": [age_r2, age_mae, age_model.alpha_],
})

age_results.to_csv(
    os.path.join(out_dir, "ica_age_prediction_summary.csv"),
    index=False,
)

age_weights = pd.DataFrame({
    "component": ica_col_names,
    "weight": age_model.coef_,
    "abs_weight": np.abs(age_model.coef_),
}).sort_values("abs_weight", ascending=False)

age_weights.to_csv(
    os.path.join(out_dir, "ica_age_prediction_weights.csv"),
    index=False,
)


#%% age trajectories for top age-related ICA components

age_plot_dir = os.path.join(out_dir, "age_trajectories")
os.makedirs(age_plot_dir, exist_ok=True)

top_age_components = age_weights["component"].head(10).tolist()

for comp in top_age_components:

    age_summary = (
        metadata_ica
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
    plt.ylabel("Mean ICA Score")
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
