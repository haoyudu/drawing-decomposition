"""
Within-Category ICA and NMF

Inputs:
  ~/Desktop/visualfeature/kiddraw_clip_embeddings.npy
  ~/Desktop/visualfeature/kiddraw_clip_metadata.parquet
  ~/Desktop/visualfeature/stringent_cleaned_dataset/<filename>   (images)

Outputs (all under within_category_analysis/outputs/):
  within_category_ica/<category>/
    S_all.npy                              ICA scores, shape (N_sample, 40)
    sample_indices.npy                     indices into the full dataset
    ica_components.npy                     mixing matrix, shape (40, 512)
    metadata_ica.parquet                   metadata + ica_000…ica_039 columns
    ica_component_top_bottom_filenames.csv top/bottom-20 filenames per component
    component_image_grids/
      ica_component_<NNN>_extremes.png     low/high row of images per component

  within_category_nmf/<category>/
    W_all.npy                              NMF scores, shape (N_sample, 40)
    sample_indices.npy
    nmf_components.npy                     basis matrix, shape (40, 512)
    metadata_nmf.parquet                   metadata + nmf_000…nmf_039 columns
    nmf_component_top_filenames.csv        top-20 filenames per component
    component_image_grids/
      nmf_component_<NNN>_top_images.png   top row of images per component
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import FastICA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
base_path = os.path.expanduser("~/Desktop/visualfeature/drawing-decomposition")
data_dir  = os.path.dirname(base_path)
image_dir = os.path.join(data_dir, "stringent_cleaned_dataset")
out_base  = os.path.join(base_path, "within_category_analysis", "outputs")

embeddings_file = os.path.join(data_dir, "kiddraw_clip_embeddings.npy")
metadata_file   = os.path.join(data_dir, "kiddraw_clip_metadata.parquet")

# ── Parameters ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    "a phone", "a cat", "a train", "a scissors", "a sheep", "a key",
    "an elephant", "a snail", "a mushroom", "a house", "a rabbit", "a boat",
]
SAMPLE_N     = 600
N_COMPONENTS = 40
RANDOM_STATE = 0
MAX_ITER_ICA = 1000
MAX_ITER_NMF = 2000
N_TOP_IMAGES = 12

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading embeddings and metadata...")
X_all    = np.load(embeddings_file)
metadata = pd.read_parquet(metadata_file).copy()

filename_col = "filename" if "filename" in metadata.columns else "image1"
metadata["clean_filename"]   = metadata[filename_col].apply(lambda x: os.path.basename(str(x)))
metadata["clean_image_path"] = metadata["clean_filename"].apply(lambda x: os.path.join(image_dir, x))

print(f"  {X_all.shape[0]} drawings, {X_all.shape[1]}-dim embeddings")

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_image_on_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def plot_ica_component_extremes(metadata_ica, component, out_dir, n_images=N_TOP_IMAGES):
    col    = f"ica_{component:03d}"
    low_df = metadata_ica.sort_values(col).head(n_images)
    hi_df  = metadata_ica.sort_values(col, ascending=False).head(n_images)

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.7, 6.0))
    plt.subplots_adjust(hspace=0.55, wspace=0.15)

    for row_i, df in enumerate([low_df, hi_df]):
        for i, (_, row) in enumerate(df.iterrows()):
            ax = axes[row_i, i]
            if os.path.exists(row["clean_image_path"]):
                ax.imshow(load_image_on_white(row["clean_image_path"]))
            ax.axis("off")
            ax.set_title(f"{row[col]:.2f}", fontsize=8)
            ax.text(0.5, -0.04, str(row.get("age", "")), ha="center", va="top",
                    transform=ax.transAxes, fontsize=8)

    axes[0, 0].set_ylabel("Low", fontsize=12)
    axes[1, 0].set_ylabel("High", fontsize=12)
    fig.suptitle(f"ICA Component {component}", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f"ica_component_{component:03d}_extremes.png"),
                dpi=300, facecolor="white")
    plt.close()


def plot_nmf_component_top_images(metadata_nmf, component, out_dir, n_images=N_TOP_IMAGES):
    col    = f"nmf_{component:03d}"
    top_df = metadata_nmf.sort_values(col, ascending=False).head(n_images)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.7, 3.2))
    if n_images == 1:
        axes = [axes]

    for i, (_, row) in enumerate(top_df.iterrows()):
        ax = axes[i]
        if os.path.exists(row["clean_image_path"]):
            ax.imshow(load_image_on_white(row["clean_image_path"]))
        ax.axis("off")
        ax.set_title(f"{row[col]:.2f}", fontsize=8)
        ax.text(0.5, -0.04, str(row.get("age", "")), ha="center", va="top",
                transform=ax.transAxes, fontsize=8)

    fig.suptitle(f"NMF Component {component}", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    plt.savefig(os.path.join(out_dir, f"nmf_component_{component:03d}_top_images.png"),
                dpi=300, facecolor="white")
    plt.close()


# ── Main loop ──────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_STATE)

for category in CATEGORIES:
    safe = category.replace(" ", "_")
    print(f"\n{'='*60}\n  {category}\n{'='*60}")

    # sample drawings
    cat_indices  = np.where((metadata["category"] == category).values)[0]
    n_available  = len(cat_indices)
    sample_size  = min(SAMPLE_N, n_available)
    sample_local = np.sort(rng.choice(n_available, size=sample_size, replace=False))
    sample_idx   = cat_indices[sample_local]

    X_cat    = X_all[sample_idx].astype(np.float64)
    meta_cat = metadata.iloc[sample_idx].reset_index(drop=True)
    print(f"  Sampled {sample_size}/{n_available}")

    # ── ICA ───────────────────────────────────────────────────────────────────
    ica_out  = os.path.join(out_base, "within_category_ica", safe)
    grid_dir = os.path.join(ica_out, "component_image_grids")
    os.makedirs(grid_dir, exist_ok=True)

    X_ica = StandardScaler().fit_transform(X_cat)
    ica   = FastICA(n_components=N_COMPONENTS, random_state=RANDOM_STATE,
                    max_iter=MAX_ITER_ICA, whiten="unit-variance")
    S_all = ica.fit_transform(X_ica)
    print(f"  ICA done — components: {ica.components_.shape}")

    np.save(os.path.join(ica_out, "S_all.npy"),         S_all)
    np.save(os.path.join(ica_out, "sample_indices.npy"), sample_idx)
    np.save(os.path.join(ica_out, "ica_components.npy"), ica.components_)

    ica_cols     = [f"ica_{i:03d}" for i in range(N_COMPONENTS)]
    metadata_ica = pd.concat([meta_cat, pd.DataFrame(S_all, columns=ica_cols)], axis=1)
    metadata_ica.to_parquet(os.path.join(ica_out, "metadata_ica.parquet"))

    extreme_rows = []
    for comp in range(N_COMPONENTS):
        col = f"ica_{comp:03d}"
        for direction, df in [("low",  metadata_ica.sort_values(col).head(20)),
                               ("high", metadata_ica.sort_values(col, ascending=False).head(20))]:
            for rank, (_, row) in enumerate(df.iterrows(), 1):
                extreme_rows.append({"component": col, "direction": direction, "rank": rank,
                                     "score": row[col], "filename": row["clean_filename"],
                                     "image_path": row["clean_image_path"],
                                     "age_numeric": row.get("age_numeric", None)})
    pd.DataFrame(extreme_rows).to_csv(
        os.path.join(ica_out, "ica_component_top_bottom_filenames.csv"), index=False)

    for comp in range(N_COMPONENTS):
        plot_ica_component_extremes(metadata_ica, comp, grid_dir)
    print(f"  ICA → {ica_out}/")

    # ── NMF ───────────────────────────────────────────────────────────────────
    nmf_out  = os.path.join(out_base, "within_category_nmf", safe)
    grid_dir = os.path.join(nmf_out, "component_image_grids")
    os.makedirs(grid_dir, exist_ok=True)

    X_nmf = np.clip(MinMaxScaler().fit_transform(X_cat), 0, None)
    nmf   = NMF(n_components=N_COMPONENTS, init="nndsvda", random_state=RANDOM_STATE,
                max_iter=MAX_ITER_NMF, solver="cd", beta_loss="frobenius",
                alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)
    W_all = nmf.fit_transform(X_nmf)
    print(f"  NMF done — reconstruction error: {nmf.reconstruction_err_:.4f}")

    np.save(os.path.join(nmf_out, "W_all.npy"),          W_all)
    np.save(os.path.join(nmf_out, "sample_indices.npy"),  sample_idx)
    np.save(os.path.join(nmf_out, "nmf_components.npy"),  nmf.components_)

    nmf_cols     = [f"nmf_{i:03d}" for i in range(N_COMPONENTS)]
    metadata_nmf = pd.concat([meta_cat, pd.DataFrame(W_all, columns=nmf_cols)], axis=1)
    metadata_nmf.to_parquet(os.path.join(nmf_out, "metadata_nmf.parquet"))

    top_rows = []
    for comp in range(N_COMPONENTS):
        col    = f"nmf_{comp:03d}"
        top_df = metadata_nmf.sort_values(col, ascending=False).head(20)
        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            top_rows.append({"component": col, "rank": rank, "score": row[col],
                             "filename": row["clean_filename"],
                             "image_path": row["clean_image_path"],
                             "age_numeric": row.get("age_numeric", None)})
    pd.DataFrame(top_rows).to_csv(
        os.path.join(nmf_out, "nmf_component_top_filenames.csv"), index=False)

    for comp in range(N_COMPONENTS):
        plot_nmf_component_top_images(metadata_nmf, comp, grid_dir)
    print(f"  NMF → {nmf_out}/")

print("\nDone. All categories processed.")
