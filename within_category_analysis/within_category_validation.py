"""
Within-Category Validation: Age Prediction + Cross-Method Agreement

1. Age prediction (RidgeCV, 5-fold CV)
   Predicts age_numeric from component scores within each category.
   ICA: 40 components | NMF: 40 components | SAE: all 4096 features (Ridge handles sparsity)
   Reports R² and MAE per category per method.

2. Cross-method agreement (Spearman on pairwise cosine similarity)
   For the same 600 drawings, compute 600×600 pairwise cosine similarity
   under each method's representation, then Spearman-correlate the upper triangles.
   Three pairs: ICA–NMF, ICA–SAE, NMF–SAE.

Inputs:
  within_category_ica/<cat>/metadata_ica.parquet
  within_category_ica/<cat>/sample_indices.npy
  within_category_nmf/<cat>/metadata_nmf.parquet
  sae_outputs/holdout_latents.npy + holdout_metadata.parquet   (phone / cat / train)
  sae_outputs/final_latents.npy                                (all other categories)
  kiddraw_clip_metadata.parquet

Outputs (all under within_category_analysis/outputs/validation/):
  age_prediction_results.csv
  age_prediction_r2.png
  cross_method_agreement.csv
  cross_method_agreement.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────────
base_path = os.path.expanduser("~/Desktop/visualfeature/drawing-decomposition")
data_dir  = os.path.dirname(base_path)
out_base  = os.path.join(base_path, "within_category_analysis", "outputs")
val_dir   = os.path.join(out_base, "validation")
os.makedirs(val_dir, exist_ok=True)

HOLDOUT_CATEGORIES = ["a phone", "a cat", "a train"]
CATEGORIES = [
    "a phone", "a cat", "a train", "a scissors", "a sheep", "a key",
    "an elephant", "a snail", "a mushroom", "a house", "a rabbit", "a boat",
]
N_COMPONENTS = 40
CV_FOLDS     = 5
RIDGE_ALPHAS = np.logspace(-3, 3, 20)

# ── Load SAE latents once ──────────────────────────────────────────────────────
print("Loading SAE latents...")
Z_holdout    = np.load(os.path.join(base_path, "sae_outputs", "holdout_latents.npy"))
meta_holdout = pd.read_parquet(os.path.join(base_path, "sae_outputs", "holdout_metadata.parquet"))
Z_final      = np.load(os.path.join(base_path, "sae_outputs", "final_latents.npy"))
meta_full    = pd.read_parquet(os.path.join(data_dir, "kiddraw_clip_metadata.parquet"))

# Build lookup: global index → row in holdout_latents
holdout_global_idx = np.where(
    meta_full["category"].isin(HOLDOUT_CATEGORIES).values
)[0]  # sorted, length 3973

print(f"  holdout latents : {Z_holdout.shape}")
print(f"  final latents   : {Z_final.shape}")

# ── Helper: upper triangle of a square matrix ──────────────────────────────────
def upper_tri(M):
    i, j = np.triu_indices(M.shape[0], k=1)
    return M[i, j]

# ── Helper: pairwise cosine similarity ────────────────────────────────────────
def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    Xn = X / norms
    return Xn @ Xn.T

# ── Helper: RidgeCV age prediction (5-fold CV) ────────────────────────────────
def predict_age_cv(X, y):
    model   = RidgeCV(alphas=RIDGE_ALPHAS, cv=CV_FOLDS)
    y_pred  = cross_val_predict(model, X, y, cv=CV_FOLDS)
    r2      = r2_score(y, y_pred)
    mae     = mean_absolute_error(y, y_pred)
    # fit on full data to get best alpha
    model.fit(X, y)
    return r2, mae, model.alpha_

# ── Helper: load ICA data for a category ──────────────────────────────────────
def load_ica(safe):
    path = os.path.join(out_base, "within_category_ica", safe, "metadata_ica.parquet")
    m    = pd.read_parquet(path)
    cols = [f"ica_{i:03d}" for i in range(N_COMPONENTS)]
    return m[cols].values, m["age_numeric"].values

# ── Helper: load NMF data for a category ──────────────────────────────────────
def load_nmf(safe):
    path = os.path.join(out_base, "within_category_nmf", safe, "metadata_nmf.parquet")
    m    = pd.read_parquet(path)
    cols = [f"nmf_{i:03d}" for i in range(N_COMPONENTS)]
    return m[cols].values, m["age_numeric"].values

# ── Helper: load SAE data aligned to ICA sample ───────────────────────────────
def load_sae(category, safe):
    sample_idx = np.load(
        os.path.join(out_base, "within_category_ica", safe, "sample_indices.npy")
    )
    if category in HOLDOUT_CATEGORIES:
        local = np.searchsorted(holdout_global_idx, sample_idx)
        Z     = Z_holdout[local]
        y     = meta_holdout.iloc[local]["age_numeric"].values
    else:
        Z = Z_final[sample_idx]
        y = meta_full.iloc[sample_idx]["age_numeric"].values
    return Z, y

# ── Main loop ──────────────────────────────────────────────────────────────────
age_rows   = []
agree_rows = []

for category in CATEGORIES:
    safe = category.replace(" ", "_")
    print(f"\n{'='*55}\n  {category}\n{'='*55}")

    X_ica, y_ica = load_ica(safe)
    X_nmf, y_nmf = load_nmf(safe)
    X_sae, y_sae = load_sae(category, safe)

    N = len(y_ica)
    print(f"  N={N}  age range {y_ica.min()}–{y_ica.max()}")

    # ── 1. Age prediction ──────────────────────────────────────────────────────
    for method, X, y in [("ICA", X_ica, y_ica),
                          ("NMF", X_nmf, y_nmf),
                          ("SAE", X_sae, y_sae)]:
        r2, mae, alpha = predict_age_cv(X, y)
        age_rows.append({
            "category": category, "method": method,
            "r2": round(r2, 4), "mae": round(mae, 4),
            "ridge_alpha": alpha, "n": N,
        })
        print(f"  {method}: R²={r2:.3f}  MAE={mae:.3f}  α={alpha:.3g}")

    # ── 2. Cross-method agreement ──────────────────────────────────────────────
    sim_ica = upper_tri(cosine_sim_matrix(X_ica))
    sim_nmf = upper_tri(cosine_sim_matrix(X_nmf))
    sim_sae = upper_tri(cosine_sim_matrix(X_sae))

    for pair, s1, s2 in [("ICA–NMF", sim_ica, sim_nmf),
                          ("ICA–SAE", sim_ica, sim_sae),
                          ("NMF–SAE", sim_nmf, sim_sae)]:
        r, p = spearmanr(s1, s2)
        agree_rows.append({
            "category": category, "pair": pair,
            "spearman_r": round(r, 4), "p_value": round(p, 6),
        })
        print(f"  {pair}: ρ={r:.3f}  p={p:.2e}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
age_df   = pd.DataFrame(age_rows)
agree_df = pd.DataFrame(agree_rows)

age_df.to_csv(os.path.join(val_dir, "age_prediction_results.csv"), index=False)
agree_df.to_csv(os.path.join(val_dir, "cross_method_agreement.csv"), index=False)
print(f"\nCSVs saved to {val_dir}/")

# ── Plot 1: Age prediction R² grouped bar chart ────────────────────────────────
methods  = ["ICA", "NMF", "SAE"]
colors   = {"ICA": "#4878CF", "NMF": "#6ACC65", "SAE": "#D65F5F"}
cats     = CATEGORIES
short    = [c.replace("a ", "").replace("an ", "") for c in cats]
x        = np.arange(len(cats))
width    = 0.26

fig, ax = plt.subplots(figsize=(14, 5))
for i, method in enumerate(methods):
    vals = [age_df.loc[(age_df.category == c) & (age_df.method == method), "r2"].iloc[0]
            for c in cats]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=method, color=colors[method], alpha=0.88)
    for bar, v in zip(bars, vals):
        if v > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("R² (5-fold CV)", fontsize=10)
ax.set_title("Age prediction from within-category component scores\n"
             "(RidgeCV, 5-fold CV)", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(min(age_df.r2.min() - 0.02, -0.05), age_df.r2.max() + 0.07)
plt.tight_layout()
plt.savefig(os.path.join(val_dir, "age_prediction_r2.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved age_prediction_r2.png")

# ── Plot 2: Cross-method agreement heatmap ────────────────────────────────────
pairs = ["ICA–NMF", "ICA–SAE", "NMF–SAE"]
mat   = np.array([[agree_df.loc[(agree_df.category == c) & (agree_df.pair == p),
                                "spearman_r"].iloc[0]
                   for p in pairs] for c in cats])

fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pairs, fontsize=10)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(short, fontsize=9)
for i in range(len(cats)):
    for j in range(len(pairs)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black" if 0.3 < mat[i, j] < 0.85 else "white")
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Spearman ρ", fontsize=9)
ax.set_title("Cross-method agreement\n(pairwise cosine similarity, upper triangle)",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(val_dir, "cross_method_agreement.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved cross_method_agreement.png")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("AGE PREDICTION R²")
pivot = age_df.pivot(index="category", columns="method", values="r2")
pivot = pivot.loc[CATEGORIES, methods]
print(pivot.to_string())

print(f"\n{'─'*60}")
print("CROSS-METHOD AGREEMENT (Spearman ρ)")
pivot2 = agree_df.pivot(index="category", columns="pair", values="spearman_r")
pivot2 = pivot2.loc[CATEGORIES, pairs]
print(pivot2.to_string())

print("\nDone.")
