# -*- coding: utf-8 -*-
"""
Run NMF on Kiddraw CLIP embeddings (full dataset, no train/test split).

NMF preprocessing matches geometry_analysis.build_representations:
MinMaxScaler -> clip non-negative.

Author: L. Kemmelmeier
"""

#%% imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


#%% paths and parameters

base_path = os.path.expanduser("~/Documents/GitHub/drawing-decomposition")

data_dir = os.path.join(base_path, "data")
image_dir = os.path.join(data_dir, "stringent_cleaned_dataset")
out_dir = os.path.join(base_path, "nmf_outputs")

os.makedirs(out_dir, exist_ok=True)

embeddings_file = os.path.join(data_dir, "kiddraw_clip_embeddings.npy")
metadata_file = os.path.join(data_dir, "kiddraw_clip_metadata.parquet")

n_components = 55
random_state = 0
max_iter = 2000
n_top_images = 20


#%% load data

X = np.load(embeddings_file)
metadata = pd.read_parquet(metadata_file).copy()

assert "category" in metadata.columns, "metadata must contain a category column"
assert "age_numeric" in metadata.columns, "metadata must contain an age_numeric column"

filename_col = "filename" if "filename" in metadata.columns else "image1"

metadata["clean_filename"] = metadata[filename_col].apply(
    lambda x: os.path.basename(str(x))
)

metadata["clean_image_path"] = metadata["clean_filename"].apply(
    lambda x: os.path.join(image_dir, x)
)

print(f"Loaded {X.shape[0]} drawings with {X.shape[1]} CLIP features")


#%% scale data for NMF

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_scaled = np.clip(X_scaled, 0, None)


#%% fit NMF

nmf = NMF(
    n_components=n_components,
    init="nndsvda",
    random_state=random_state,
    max_iter=max_iter,
    solver="cd",
    beta_loss="frobenius",
    alpha_W=0.0,
    alpha_H=0.0,
    l1_ratio=0.0,
)

W = nmf.fit_transform(X_scaled)

print(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")


#%% save NMF outputs

np.save(os.path.join(out_dir, "W.npy"), W)

np.save(os.path.join(out_dir, "nmf_components.npy"), nmf.components_)

metadata.to_parquet(os.path.join(out_dir, "metadata.parquet"))


#%% attach NMF scores to metadata

nmf_col_names = [f"nmf_{i:03d}" for i in range(n_components)]

W_df = pd.DataFrame(W, columns=nmf_col_names)

metadata_nmf = pd.concat(
    [metadata.reset_index(drop=True), W_df],
    axis=1,
)

metadata_nmf.to_parquet(os.path.join(out_dir, "metadata_nmf.parquet"))


#%% save top image names for each NMF component

component_top_rows = []

for component in range(n_components):

    component_col = f"nmf_{component:03d}"

    top_df = (
        metadata_nmf
        .sort_values(component_col, ascending=False)
        .head(n_top_images)
    )

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):

        component_top_rows.append({
            "component": component_col,
            "rank": rank,
            "score": row[component_col],
            "filename": row["clean_filename"],
            "image_path": row["clean_image_path"],
            "category": row["category"],
            "age_numeric": row["age_numeric"],
        })

component_top_df = pd.DataFrame(component_top_rows)

component_top_df.to_csv(
    os.path.join(out_dir, "nmf_component_top_filenames.csv"),
    index=False,
)


#%% plot top images for each NMF component

component_plot_dir = os.path.join(out_dir, "component_image_grids")
os.makedirs(component_plot_dir, exist_ok=True)


def load_image_on_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img)
    return img.convert("RGB")


def plot_component_top_images(metadata_nmf, component, n_images=12):

    component_col = f"nmf_{component:03d}"

    top_df = (
        metadata_nmf
        .sort_values(component_col, ascending=False)
        .head(n_images)
    )

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(n_images * 1.7, 3.2),
    )

    if n_images == 1:
        axes = [axes]

    for i, (_, row) in enumerate(top_df.iterrows()):

        img = load_image_on_white(row["clean_image_path"])
        axes[i].imshow(img)
        axes[i].axis("off")

        axes[i].set_title(
            f"{row[component_col]:.2f}",
            fontsize=8,
        )

        axes[i].text(
            0.5,
            -0.04,
            str(row["category"]),
            ha="center",
            va="top",
            transform=axes[i].transAxes,
            fontsize=8,
        )

    fig.suptitle(
        f"NMF Component {component}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])

    save_path = os.path.join(
        component_plot_dir,
        f"nmf_component_{component:03d}_top_images.png",
    )

    plt.savefig(save_path, dpi=300, facecolor="white")
    plt.close()


for component in range(n_components):
    plot_component_top_images(
        metadata_nmf=metadata_nmf,
        component=component,
        n_images=12,
    )


#%% category-component means

category_component_means = (
    metadata_nmf
    .groupby("category")[nmf_col_names]
    .mean()
)

category_component_means.to_csv(
    os.path.join(out_dir, "category_component_mean_scores.csv"),
)


#%% helper function for ordering heatmaps

def order_category_component_matrix(df, component_prefix="nmf_"):

    ranking_values = df.copy()

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


#%% helper function for plotting heatmaps

def plot_category_component_heatmap(
    heatmap_df,
    component_labels,
    save_path,
    colorbar_label,
    vmin=None,
    vmax=None,
):

    fig, ax = plt.subplots(figsize=(22, 18))

    im = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
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
        "NMF Component",
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
        colorbar_label,
        fontsize=20,
        fontweight="bold",
        labelpad=15,
    )

    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


#%% raw category-component heatmap

category_heatmap_dir = os.path.join(
    out_dir,
    "category_component_heatmaps",
)

os.makedirs(category_heatmap_dir, exist_ok=True)

raw_heatmap_df, raw_component_labels = order_category_component_matrix(
    category_component_means,
    component_prefix="nmf_",
)

plot_category_component_heatmap(
    heatmap_df=raw_heatmap_df,
    component_labels=raw_component_labels,
    save_path=os.path.join(
        category_heatmap_dir,
        "category_component_heatmap.png",
    ),
    colorbar_label="Mean NMF Score",
)


#%% within-component normalized heatmap

normalized_heatmap_dir = os.path.join(
    out_dir,
    "category_component_heatmaps_normalized",
)

os.makedirs(normalized_heatmap_dir, exist_ok=True)

normalized_category_component_means = (
    category_component_means - category_component_means.min(axis=0)
) / (
    category_component_means.max(axis=0)
    - category_component_means.min(axis=0)
    + 1e-10
)

normalized_heatmap_df, normalized_component_labels = order_category_component_matrix(
    normalized_category_component_means,
    component_prefix="nmf_",
)

plot_category_component_heatmap(
    heatmap_df=normalized_heatmap_df,
    component_labels=normalized_component_labels,
    save_path=os.path.join(
        normalized_heatmap_dir,
        "category_component_heatmap_normalized.png",
    ),
    colorbar_label="Normalized Component Score",
    vmin=0,
    vmax=1,
)


#%% category selectivity table

selectivity_dir = os.path.join(out_dir, "category_selectivity")
os.makedirs(selectivity_dir, exist_ok=True)

selectivity_rows = []

for comp in nmf_col_names:

    means = category_component_means[comp]

    top_category = means.idxmax()
    top_score = means.max()
    mean_other_score = means.drop(top_category).mean()

    selectivity_rows.append({
        "component": comp,
        "top_category": top_category,
        "top_score": top_score,
        "mean_other_score": mean_other_score,
        "selectivity_score": top_score - mean_other_score,
    })

selectivity_df = (
    pd.DataFrame(selectivity_rows)
    .sort_values("selectivity_score", ascending=False)
)

selectivity_df.to_csv(
    os.path.join(
        selectivity_dir,
        "nmf_component_category_selectivity.csv",
    ),
    index=False,
)


#%% top categories for each component

top_categories_dir = os.path.join(out_dir, "top_categories_per_component")
os.makedirs(top_categories_dir, exist_ok=True)

top_category_rows = []

for comp in nmf_col_names:

    top_categories = (
        category_component_means[comp]
        .sort_values(ascending=False)
        .head(10)
    )

    for rank, (category, score) in enumerate(top_categories.items(), start=1):

        top_category_rows.append({
            "component": comp,
            "rank": rank,
            "category": category,
            "mean_score": score,
        })

top_categories_df = pd.DataFrame(top_category_rows)

top_categories_df.to_csv(
    os.path.join(
        top_categories_dir,
        "top_categories_per_component.csv",
    ),
    index=False,
)


#%% age trajectories for components most correlated with age

age_plot_dir = os.path.join(out_dir, "age_trajectories")
os.makedirs(age_plot_dir, exist_ok=True)

age_component_corrs = []

for comp in nmf_col_names:

    r = metadata_nmf[[comp, "age_numeric"]].corr().iloc[0, 1]

    age_component_corrs.append({
        "component": comp,
        "age_corr": r,
        "abs_age_corr": abs(r),
    })

age_component_corrs_df = (
    pd.DataFrame(age_component_corrs)
    .sort_values("abs_age_corr", ascending=False)
)

age_component_corrs_df.to_csv(
    os.path.join(out_dir, "age_component_correlations.csv"),
    index=False,
)

top_age_components = age_component_corrs_df["component"].head(10).tolist()

for comp in top_age_components:

    age_summary = (
        metadata_nmf
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
    plt.ylabel("Mean NMF Score")
    plt.title(f"Age trajectory: {comp}")
    plt.tight_layout()

    save_path = os.path.join(
        age_plot_dir,
        f"{comp}_age_trajectory.png",
    )

    plt.savefig(save_path, dpi=300)
    plt.close()


#%% done

print(f"Done. Outputs saved to: {out_dir}")
