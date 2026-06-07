# drawing-decomposition

Decomposes CLIP embeddings of ~31k children's drawings using a TopK Sparse Autoencoder (SAE) and compares it against classical methods (PCA, ICA, NMF). The central question is whether unsupervised decomposition recovers interpretable, human-readable visual features — and whether it reorganizes the representational geometry of the embedding space or merely redescribes it.

**Context:** CLIP embeddings are dense 512-dimensional vectors where category information is already encoded (CLIP was trained on internet image-text pairs). The SAE maps each drawing to a sparse 4096-dimensional code with exactly k=32 active features, trading ambient dimensionality for sparsity to combat entanglement. All decompositions are fit and evaluated on held-out val folds so the SAE is assessed out-of-sample.

## Files

| Script | Purpose |
|---|---|
| `model.py` | TopK SAE architecture ([OpenAI sparse_autoencoder](https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py)) |
| `train_sae.py` | Train SAE with k-fold cross-validation and AuxK loss |
| `evaluate_sae.py` | Sanity-check a trained fold: reconstruction R², dead features, category probe |
| `visualize_features.py` | Save top-activating drawing grids per feature |
| `heatmap_features.py` | Category × feature mean-activation heatmap + per-feature collages |
| `run_ica.py` | Fit FastICA on CLIP embeddings; save components and per-category heatmaps |
| `run_nmf.py` | Fit NMF on CLIP embeddings; save components and per-category heatmaps |
| `geometry_analysis.py` | Compare representational geometry: PCA scree + within/between-category RDM correlations across all four methods |

## Environment

```bash
conda activate vislearnlabpy
```

Requires PyTorch (CUDA), NumPy, pandas, scikit-learn, seaborn, tqdm. Training uses GPU automatically if available.

## Data

Place in `data/`:
- `kiddraw_clip_embeddings.npy` — shape `(N, 512)`, CLIP ViT last-layer embeddings
- `kiddraw_clip_metadata.parquet` — one row per drawing; must include `filename`, `category`, `age` columns

## Workflow

### 1. Train

```bash
# Full 5-fold run (all defaults)
python train_sae.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy

# Quick sanity check (1 fold, 50 epochs)
python train_sae.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy \
    --n_folds 1 --num_epochs 50
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--k` | `32` | TopK sparsity (active features per sample) |
| `--M` | `4096` | Number of learned SAE features |
| `--lr` | `1e-4` | Adam learning rate |
| `--num_epochs` | `200` | Epochs per fold |
| `--auxk` | `None` | AuxK dead-feature loss (recommended: `k//2 = 16`) |
| `--auxk_coef` | `1/32` | Weight on AuxK loss |
| `--output_dir` | `./outputs` | Where to save models and loss CSVs |

Outputs per fold `i`: `outputs/fold{i}_sae.pt`, `outputs/fold{i}_losses.csv`

---

### 2. Evaluate

```bash
python evaluate_sae.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy \
    --metadata_path   data/kiddraw_clip_metadata.parquet \
    --model_path      outputs/fold2_sae.pt
```

Prints: reconstruction R² vs. PCA baseline, dead-feature fraction, activation histogram, category probe accuracy (CLIP embeddings vs. SAE codes).

Add `--skip_probe` to skip the slow logistic regression step.

---

### 3. Visualize features

```bash
# Auto-select 10 most-used features
python visualize_features.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy \
    --metadata_path   data/kiddraw_clip_metadata.parquet \
    --model_path      outputs/fold2_sae.pt \
    --fold 2

# Specific feature indices
python visualize_features.py ... --features 42 137 801
```

Saves a grid PNG per feature to `outputs/feature_viz/`.

---

### 4. Category × feature heatmap

```bash
python heatmap_features.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy \
    --metadata_path   data/kiddraw_clip_metadata.parquet \
    --model_path      outputs/fold2_sae.pt \
    --fold 2 \
    --n_features 50 \
    --zscore \
    --output_path outputs/category_feature_heatmap.png

# Also save per-feature collages (top 5 drawings from primary category)
python heatmap_features.py ... --save_top_images --top_k 5
```

Selects the `--n_features` most category-selective features, orders rows and columns so high values fall near the diagonal, and saves a heatmap PNG. With `--save_top_images`, also saves one collage PNG per feature to `outputs/category_feature_heatmap_collages/`.

---

### 5. ICA and NMF baselines

```bash
python run_ica.py   # outputs to ica_outputs/
python run_nmf.py   # outputs to nmf_outputs/
```

Hardcoded to 50 components. Each script saves component scores, per-category mean heatmaps, top/bottom activating drawings per component, and age trajectory plots. Preprocessing: ICA uses `StandardScaler`; NMF uses `MinMaxScaler` + clip to non-negative.

---

### 6. Geometry analysis

Answers: does decomposition reorganize the CLIP embedding space, or merely redescribe it?

```bash
python geometry_analysis.py \
    --embeddings_path data/kiddraw_clip_embeddings.npy \
    --metadata_path   data/kiddraw_clip_metadata.parquet \
    --model_path      outputs/fold2_sae.pt \
    --fold 2 \
    --min_cat_size 30 \
    --output_path outputs/geometry_analysis.svg
```

Two panels saved as SVG:
- **Panel A** — PCA scree with markers at 80/90/95% explained variance; reports the knee dimensionality used for PCA/ICA/NMF
- **Panel B** — Grouped bar chart: within-category and between-category Spearman RDM correlation (vs. full CLIP space) for each method

All four methods are fit and evaluated on the held-out val fold only. Between-category correlations serve as a sanity check (expected ≥ 0.95 for all methods); within-category correlations are the informative result.
