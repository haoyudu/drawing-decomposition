"""
geometry_analysis.py — does decomposition reorganize the CLIP embedding space,
or merely redescribe it?

two analyses:
  1. compressibility (scree): pca cumulative explained variance → knee dimensionality
  2. geometry preservation (rdm correlation): within- and between-category spearman
     correlation of pairwise cosine dissimilarity matrices across four methods
     (pca, ica, nmf, sae)

methods note (for reporting):
  all four decompositions — pca, ica, nmf, and sae — are fit and evaluated
  exclusively on the held-out val fold for the specified --fold. the sae was
  trained on the complementary training folds and is therefore assessed
  out-of-sample; the classical methods (pca, ica, nmf) are also fit only on
  the val fold drawings to ensure a fair comparison on the same drawing set.

usage:
  python geometry_analysis.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold2_sae.pt \
      --fold 2
"""

import argparse
import os

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from model import Autoencoder, TopK


# ── fold helpers (matches visualize_features.py / heatmap_features.py) ─────────

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


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# analysis 1: scree / compressibility

def pca_scree(X):
    """fit full pca on X; return cumulative explained variance and threshold counts."""
    pca = PCA().fit(X)
    cev = np.cumsum(pca.explained_variance_ratio_)

    thresholds = {}
    for t in (0.80, 0.90, 0.95):
        thresholds[t] = int(np.searchsorted(cev, t)) + 1

    knee_n = thresholds[0.90]  # default: 90% threshold

    # kneed as cross-check only; not a hard dependency
    try:
        from kneed import KneeLocator
        kl = KneeLocator(
            np.arange(1, len(cev) + 1), cev,
            curve="concave", direction="increasing",
        )
        if kl.knee is not None:
            print(f"  kneed knee: {kl.knee}  "
                  f"(using 90% threshold = {knee_n} as default)")
    except ImportError:
        pass

    return cev, thresholds, knee_n


# build reduced representations

def build_representations(X, n_components, seed):
    """
    fit pca, ica, nmf on X and return dict of reduced arrays (same row order as X).

    preprocessing follows existing scripts:
      ica — StandardScaler then FastICA(whiten="unit-variance")  [run_ica.py]
      nmf — MinMaxScaler then clip(0, inf)                       [run_nmf.py]
    """
    reps = {}

    # pca
    pca = PCA(n_components=n_components, random_state=seed)
    reps["PCA"] = pca.fit_transform(X)

    # ica on pca-reduced data: avoids fastica's expensive internal full svd
    scaler_ica = StandardScaler()
    X_ica = scaler_ica.fit_transform(X)
    X_ica_pca = PCA(n_components=n_components, random_state=seed).fit_transform(X_ica)
    ica = FastICA(n_components=n_components, random_state=seed,
                  max_iter=200, whiten=False)
    reps["ICA"] = ica.fit_transform(X_ica_pca)

    # nmf
    scaler_nmf = MinMaxScaler()
    X_nmf = np.clip(scaler_nmf.fit_transform(X), 0, None)
    nmf = NMF(n_components=n_components, init="nndsvda",
              random_state=seed, max_iter=2000,
              solver="cd", beta_loss="frobenius")
    reps["NMF"] = nmf.fit_transform(X_nmf)

    return reps


# analysis 2: rdm correlation

def cosine_rdm(R):
    """flattened upper triangle of pairwise cosine distance matrix."""
    return pdist(R, metric="cosine")


def within_category_rdm(X, R_dict, categories, cat_labels, max_per_cat, n_seeds):
    """
    for each category: correlate (spearman) the full-space cosine rdm with
    each reduced repr's cosine rdm, computed on drawings within that category.
    categories exceeding max_per_cat are subsampled; results averaged over n_seeds.

    returns dict: method → list of per-category r values.
    """
    results = {m: [] for m in R_dict}

    for cat in cat_labels:
        idx = np.where(categories == cat)[0]

        if len(idx) > max_per_cat:
            cat_rs = {m: [] for m in R_dict}
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                sub = rng.choice(idx, size=max_per_cat, replace=False)
                rdm_full = cosine_rdm(X[sub])
                for m, R in R_dict.items():
                    r, _ = spearmanr(rdm_full, cosine_rdm(R[sub]))
                    cat_rs[m].append(r)
            for m in R_dict:
                results[m].append(float(np.mean(cat_rs[m])))
        else:
            rdm_full = cosine_rdm(X[idx])
            for m, R in R_dict.items():
                r, _ = spearmanr(rdm_full, cosine_rdm(R[idx]))
                results[m].append(r)

    return results


def between_category_rdm(X, R_dict, categories, cat_labels):
    """centroid-based rdm correlation; one spearman r per method."""
    centroids_full = np.stack([X[categories == c].mean(0) for c in cat_labels])
    rdm_full = cosine_rdm(centroids_full)
    results = {}
    for m, R in R_dict.items():
        centroids_r = np.stack([R[categories == c].mean(0) for c in cat_labels])
        r, _ = spearmanr(rdm_full, cosine_rdm(centroids_r))
        results[m] = r
    return results


# plotting

def plot_results(cev, thresholds, knee_n, within_res, between_res,
                 methods, output_path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # panel a: scree
    ax_a.plot(np.arange(1, len(cev) + 1), cev, lw=1.5, color="steelblue")
    thresh_style = {0.80: ("orange", "--"), 0.90: ("green", "--"), 0.95: ("red", "--")}
    for t, (col, ls) in thresh_style.items():
        n = thresholds[t]
        ax_a.axvline(n, color=col, linestyle=ls, lw=1,
                     label=f"{int(t*100)}%  →  {n} components")
    ax_a.axvline(knee_n, color="black", linestyle=":", lw=1.5,
                 label=f"knee (90%) = {knee_n}")
    ax_a.set_xlabel("number of components")
    ax_a.set_ylabel("cumulative explained variance")
    ax_a.set_title("A   compressibility (PCA scree)")
    ax_a.legend(fontsize=8)
    ax_a.set_xlim(0, len(cev))
    ax_a.set_ylim(0, 1.02)

    # panel b: grouped bar chart
    x = np.arange(len(methods))
    width = 0.35
    within_means = [np.mean(within_res[m]) for m in methods]
    within_sds   = [np.std(within_res[m])  for m in methods]
    between_vals = [between_res[m]          for m in methods]

    ax_b.bar(x - width / 2, within_means, width, yerr=within_sds, capsize=4,
             label="within-category", color="steelblue", alpha=0.85)
    ax_b.bar(x + width / 2, between_vals, width,
             label="between-category", color="coral", alpha=0.85)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods)
    ax_b.set_ylabel("Spearman r  (vs. full CLIP space)")
    ax_b.set_title("B   geometry preservation (RDM correlation)")
    ax_b.set_ylim(0, 1.08)
    ax_b.axhline(1.0, color="gray", lw=0.8, linestyle="--")
    ax_b.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="geometry analysis: compressibility + rdm correlation")
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--metadata_path",   required=True)
    parser.add_argument("--model_path",      required=True)
    parser.add_argument("--fold",         type=int, default=0)
    parser.add_argument("--n_folds",      type=int, default=5)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--k",            type=int, default=32)
    parser.add_argument("--M",            type=int, default=4096)
    parser.add_argument("--batch_size",   type=int, default=512)
    parser.add_argument("--category_col", default="category")
    parser.add_argument("--n_components", type=int, default=None,
                        help="components for pca/ica/nmf (default: pca 90%% threshold)")
    parser.add_argument("--max_per_cat",  type=int, default=800,
                        help="subsample threshold for within-category rdm")
    parser.add_argument("--min_cat_size", type=int, default=15,
                        help="skip categories with fewer val drawings than this")
    parser.add_argument("--n_seeds",      type=int, default=5,
                        help="seeds for subsampled-category averaging")
    parser.add_argument("--output_path",  default="outputs/geometry_analysis.svg")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load data
    X_np = np.load(args.embeddings_path).astype(np.float32)
    N, D = X_np.shape
    meta = pd.read_parquet(args.metadata_path)
    assert len(meta) == N

    # restrict to val fold — all four methods are fit and evaluated here only,
    # so the sae (trained on complementary folds) is assessed out-of-sample
    # alongside pca, ica, and nmf
    val_idx  = get_val_indices(N, args.n_folds, args.fold, args.seed)
    X_val    = X_np[val_idx]
    meta_val = meta.iloc[val_idx].reset_index(drop=True)
    categories = meta_val[args.category_col].values
    print(f"val fold {args.fold}: {len(val_idx)} drawings")

    # per-category n report and min-n filtering
    all_cats = sorted(np.unique(categories))
    cat_ns   = {c: int((categories == c).sum()) for c in all_cats}

    section("per-category drawing counts (val fold)")
    print(f"  {'category':<24}  {'n':>5}")
    print(f"  {'-'*32}")
    for c in all_cats:
        flag = "  [excluded]" if cat_ns[c] < args.min_cat_size else ""
        print(f"  {c:<24}  {cat_ns[c]:>5}{flag}")

    cat_labels  = [c for c in all_cats if cat_ns[c] >= args.min_cat_size]
    n_excluded  = len(all_cats) - len(cat_labels)
    print(f"\n  {len(all_cats)} categories total | "
          f"{n_excluded} excluded (n < {args.min_cat_size}) | "
          f"{len(cat_labels)} retained")

    # analysis 1: scree
    section("1. compressibility (pca scree)")
    cev, thresholds, knee_n = pca_scree(X_val)
    print(f"  80% variance: {thresholds[0.80]} components")
    print(f"  90% variance: {thresholds[0.90]} components")
    print(f"  95% variance: {thresholds[0.95]} components")
    print(f"  knee (default = 90%): {knee_n} components")

    n_components = args.n_components if args.n_components is not None else knee_n
    print(f"  → using n_components = {n_components} for pca / ica / nmf")

    # sae encoding 
    section("2. sae encoding (val fold)")
    model = Autoencoder(
        n_latents=args.M, n_inputs=D,
        activation=TopK(k=args.k), tied=False, normalize=True,
    ).to(device)
    sd = torch.load(args.model_path, map_location=device)
    # state_dict() is overridden in model.py to add non-tensor "activation" and
    # "activation_state_dict" entries; filter to tensors before loading
    sd_clean = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    assert not missing, f"missing keys in checkpoint: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    print(f"loaded: {args.model_path}")
    Z = encode_all(model, torch.from_numpy(X_val), args.batch_size, device)
    print(f"sae codes: {Z.shape}")

    # fit classical reducers on val fold 
    section("3. fitting pca / ica / nmf on val fold")
    reps = build_representations(X_val, n_components, args.seed)
    reps["SAE"] = Z
    methods = ["PCA", "ICA", "NMF", "SAE"]
    for m in methods:
        print(f"  {m}: {reps[m].shape}")

    # within-category rdm 
    section("4. within-category rdm correlation")
    print(f"  max_per_cat={args.max_per_cat}  n_seeds={args.n_seeds}  "
          f"categories={len(cat_labels)}")
    within_res = within_category_rdm(
        X_val, reps, categories, cat_labels, args.max_per_cat, args.n_seeds,
    )

    # between-category rdm 
    section("5. between-category rdm correlation (centroid-based)")
    between_res = between_category_rdm(X_val, reps, categories, cat_labels)

    # summary table
    section("summary")
    print(f"\n  pca knee: {knee_n} components  "
          f"(80%={thresholds[0.80]}, 90%={thresholds[0.90]}, 95%={thresholds[0.95]})")
    print(f"\n  {'method':<6}  {'within mean':>12}  {'within sd':>10}  "
          f"{'between':>8}  {'n_cats':>7}")
    print(f"  {'-'*56}")
    for m in methods:
        wm = np.mean(within_res[m])
        ws = np.std(within_res[m])
        bv = between_res[m]
        nc = len(within_res[m])
        print(f"  {m:<6}  {wm:>12.4f}  {ws:>10.4f}  {bv:>8.4f}  {nc:>7}")

    # plot 
    plot_results(cev, thresholds, knee_n, within_res, between_res,
                 methods, args.output_path)


if __name__ == "__main__":
    main()
