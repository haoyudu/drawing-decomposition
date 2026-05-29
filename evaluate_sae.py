"""
Sanity-check evaluation for a trained SAE fold.

Checks (in order of importance):
  1. Reconstruction quality: R², per-drawing cosine similarity, vs. PCA baseline
  2. Feature usage: dead-feature fraction, activation-frequency histogram
  3. Semantic preservation: linear probe (category accuracy on Z vs. X)

Usage:
  python evaluate_sae.py \
      --embeddings_path data/kiddraw_clip_embeddings.npy \
      --metadata_path   data/kiddraw_clip_metadata.parquet \
      --model_path      outputs/fold0_sae.pt \
      --k 32 --M 4096 \
      --category_col    category
"""

import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from model import Autoencoder, TopK


# ── helpers ────────────────────────────────────────────────────────────────────

def get_codes_and_recons(model, X_tensor, batch_size=512, device="cpu"):
    """Run the frozen encoder over all drawings; return (Z, recons) as numpy."""
    model.eval()
    Zs, Rs = [], []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            _, latents, recons = model(batch)
            Zs.append(latents.cpu().numpy())
            Rs.append(recons.cpu().numpy())
    return np.vstack(Zs), np.vstack(Rs)


def cosine_sim_rowwise(A, B):
    """Cosine similarity between matching rows of A and B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── checks ─────────────────────────────────────────────────────────────────────

def check_reconstruction(X, recons):
    section("1. Reconstruction quality")

    # MSE
    mse_sae  = ((X - recons) ** 2).mean()
    mse_mean = ((X - X.mean(axis=0, keepdims=True)) ** 2).mean()
    r2 = 1.0 - mse_sae / mse_mean
    print(f"  MSE (SAE):        {mse_sae:.6f}")
    print(f"  MSE (mean pred):  {mse_mean:.6f}")
    print(f"  R²:               {r2:.4f}   {'✓ good' if r2 > 0.7 else '✗ low — check training'}")

    # Cosine similarity
    cos = cosine_sim_rowwise(X, recons)
    print(f"  Cosine sim:  mean={cos.mean():.4f}  median={np.median(cos):.4f}"
          f"  p5={np.percentile(cos, 5):.4f}  {'✓' if cos.mean() > 0.9 else '△ lower than expected'}")

    return r2, cos.mean()


def check_pca_baseline(X, recons, k):
    section("2. PCA baseline (optimal linear reconstruction at k components)")

    # Compare at same budget as SAE: k components (same as TopK sparsity)
    # PCA with n_components=D would be perfect reconstruction (useless baseline)
    mse_mean = ((X - X.mean(axis=0, keepdims=True)) ** 2).mean()
    mse_sae  = ((X - recons) ** 2).mean()

    for n_comp in [k, k * 2, k * 4]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.inverse_transform(pca.fit_transform(X))
        mse_pca = ((X - X_pca) ** 2).mean()
        r2_pca  = 1.0 - mse_pca / mse_mean
        ratio   = mse_sae / mse_pca
        print(f"  PCA({n_comp:4d} components):  R²={r2_pca:.4f}  SAE/PCA MSE ratio={ratio:.3f}"
              f"  {'✓' if ratio < 5 else '△' if ratio < 20 else '✗'}")

    print("  (ratio ~1 = SAE matches PCA quality at that budget; <5 is acceptable)")


def check_feature_usage(Z, M):
    section("3. Feature usage / dead features")

    # Activation frequency per feature (fraction of drawings that activate it)
    act_freq = (Z > 0).mean(axis=0)   # shape (M,)

    dead_threshold = 1e-3
    n_dead  = (act_freq < dead_threshold).sum()
    n_total = M
    print(f"  Dead features (freq < {dead_threshold}): {n_dead}/{n_total}  "
          f"({100*n_dead/n_total:.1f}%)  {'✓' if n_dead/n_total < 0.5 else '✗ too many dead — consider lower M or higher k'}")

    # L0 per drawing (should ≈ k)
    l0 = (Z > 0).sum(axis=1)
    print(f"  L0 per drawing:  mean={l0.mean():.1f}  std={l0.std():.1f}  "
          f"min={l0.min()}  max={l0.max()}")

    # Histogram buckets
    buckets = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
    labels  = ["0", "1e-4", "1e-3", "1e-2", "0.1", "0.5", "1.0"]
    print("  Activation-frequency histogram:")
    for lo, hi, lab in zip(buckets[:-1], buckets[1:], labels[1:]):
        count = ((act_freq >= lo) & (act_freq < hi)).sum()
        bar   = "█" * int(30 * count / n_total)
        print(f"    freq < {lab:>5s} | {bar:<30s} {count:5d}")

    return act_freq


def check_semantic_preservation(X, Z, labels):
    section("4. Semantic preservation: category linear probe")

    le  = LabelEncoder()
    y   = le.fit_transform(labels)
    n_classes = len(le.classes_)

    X_tr, X_te, Z_tr, Z_te, y_tr, y_te = train_test_split(
        X, Z, y, test_size=0.2, random_state=42, stratify=y
    )

    def probe(feats_tr, feats_te, name):
        print(f"  Fitting probe on {name} ...", flush=True)
        clf = LogisticRegression(max_iter=300, C=1.0, solver="saga",
                                 n_jobs=-1, verbose=0)
        clf.fit(feats_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(feats_te))
        print(f"  {name:30s}  accuracy={acc:.3f}  ({n_classes} classes)")
        return acc

    acc_x = probe(X_tr, X_te, "Original CLIP embeddings (X)")
    acc_z = probe(Z_tr, Z_te, "SAE activations (Z)")

    ratio = acc_z / acc_x if acc_x > 0 else 0
    flag  = "✓ preserved" if ratio > 0.9 else ("△ partial loss" if ratio > 0.7 else "✗ significant loss")
    print(f"\n  Z/X accuracy ratio: {ratio:.3f}  {flag}")
    print("  (>0.9 = SAE codes carry as much category info as raw CLIP)")
    return acc_x, acc_z


def top_features_summary(Z, act_freq, top_n=10):
    section(f"5. Top-{top_n} most active features (visual check targets)")
    top_idx = np.argsort(act_freq)[::-1][:top_n]
    print(f"  {'Feature':>8}  {'Activation freq':>16}  {'Mean activation (when >0)':>26}")
    for f in top_idx:
        mask = Z[:, f] > 0
        mean_act = Z[mask, f].mean() if mask.sum() > 0 else 0.0
        print(f"  {f:>8d}  {act_freq[f]:>16.4f}  {mean_act:>26.4f}")
    print("\n  -> Run visualize_features.py on these feature indices to eyeball top-activating drawings.")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--metadata_path",   required=True)
    parser.add_argument("--model_path",      required=True, help="e.g. outputs/fold0_sae.pt")
    parser.add_argument("--k",  type=int, default=32)
    parser.add_argument("--M",  type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--category_col", default="category",
                        help="Column in metadata parquet with drawing category labels")
    parser.add_argument("--skip_probe", action="store_true",
                        help="Skip the slow logistic regression probe (checks 1-3 still run)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    X_np = np.load(args.embeddings_path).astype(np.float32)
    N, D = X_np.shape
    X_tensor = torch.from_numpy(X_np)
    print(f"Embeddings: ({N}, {D})")

    meta = pd.read_parquet(args.metadata_path)
    assert len(meta) == N, f"Metadata rows {len(meta)} != embedding rows {N}"
    labels = meta[args.category_col].values
    n_cat  = len(np.unique(labels))
    print(f"Categories: {n_cat}  (column='{args.category_col}')")

    # Load model
    model = Autoencoder(
        n_latents=args.M,
        n_inputs=D,
        activation=TopK(k=args.k),
        tied=False,
        normalize=True,
    ).to(device)
    sd = torch.load(args.model_path, map_location=device)
    # strip non-tensor entries that our state_dict() adds
    sd_clean = {k: v for k, v in sd.items()
                if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd_clean, strict=False)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Forward pass
    Z, recons = get_codes_and_recons(model, X_tensor, args.batch_size, device)

    # Run checks
    r2, cos_mean = check_reconstruction(X_np, recons)
    r2_pca       = check_pca_baseline(X_np, recons, args.k)
    act_freq     = check_feature_usage(Z, args.M)
    if not args.skip_probe:
        acc_x, acc_z = check_semantic_preservation(X_np, Z, labels)
    top_features_summary(Z, act_freq)

    # Summary verdict
    section("Summary")
    if args.skip_probe:
        ok = (r2 > 0.7) and (cos_mean > 0.85)
        print(f"  R²={r2:.3f}  cosine={cos_mean:.3f}  (probe skipped)")
    else:
        ok = (r2 > 0.7) and (cos_mean > 0.85) and (acc_z / max(acc_x, 1e-9) > 0.8)
        print(f"  R²={r2:.3f}  cosine={cos_mean:.3f}  probe_ratio={acc_z/max(acc_x,1e-9):.3f}")
    print(f"  Overall: {'✓ SAE looks healthy — proceed to full analysis' if ok else '✗ Address issues above before downstream analysis'}")


if __name__ == "__main__":
    main()
