import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

from model import Autoencoder, TopK


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TopK sparse autoencoder with k-fold CV")
    parser.add_argument("--embeddings_path", required=True, help=".npy file of shape (N, D)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--M", type=int, default=4096, help="Number of learned features")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./outputs")
    return parser.parse_args()


def val_mse(model, val_tensor, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for start in range(0, len(val_tensor), batch_size):
            batch = val_tensor[start : start + batch_size].to(device)
            _, _, recons = model(batch)
            total_loss += ((recons - batch) ** 2).mean().item() * len(batch)
            total_n += len(batch)
    model.train()
    return total_loss / total_n


def train_fold(fold_idx, train_tensor, val_tensor, args, device):
    D = train_tensor.shape[1]
    model = Autoencoder(
        n_latents=args.M,
        n_inputs=D,
        activation=TopK(k=args.k),
        tied=False,
        normalize=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Fold {fold_idx} | Model: Autoencoder(M={args.M}, D={D}, k={args.k}) ===")
    print(f"    Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    trajectory = []
    for epoch in tqdm(range(1, args.num_epochs + 1), desc=f"Fold {fold_idx} epochs"):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, _, recons = model(batch)
            loss = ((recons - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            # renormalize decoder columns
            model.decoder.weight.data /= (
                model.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
            )
            train_loss_sum += loss.item() * len(batch)
            train_n += len(batch)

        train_mse = train_loss_sum / train_n
        v_mse = val_mse(model, val_tensor, args.batch_size, device)
        print(f"  epoch={epoch:4d}  train_mse={train_mse:.6f}  val_mse={v_mse:.6f}")
        trajectory.append({"epoch": epoch, "train_mse": train_mse, "val_mse": v_mse})

    return model, trajectory


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    data = np.load(args.embeddings_path).astype(np.float32)
    X = torch.from_numpy(data)
    N, D = X.shape
    print(f"Loaded embeddings: shape=({N}, {D})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    indices = np.arange(N)

    if args.n_folds == 1:
        # Single 90/10 split
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(N)
        split = int(0.9 * N)
        train_idx, val_idx = perm[:split], perm[split:]
        folds = [(train_idx, val_idx)]
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        folds = list(kf.split(indices))

    final_val_mses = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_tensor = X[train_idx]
        val_tensor = X[val_idx]

        model, trajectory = train_fold(fold_i, train_tensor, val_tensor, args, device)

        # Save loss trajectory
        csv_path = os.path.join(args.output_dir, f"fold{fold_i}_losses.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
            writer.writeheader()
            writer.writerows(trajectory)
        print(f"  -> Saved losses to {csv_path}")

        # Save model
        pt_path = os.path.join(args.output_dir, f"fold{fold_i}_sae.pt")
        torch.save(model.state_dict(), pt_path)
        print(f"  -> Saved model to {pt_path}")

        final_val_mses.append(trajectory[-1]["val_mse"])

    print("\n=== Final Results ===")
    for i, v in enumerate(final_val_mses):
        print(f"  Fold {i}: val_mse={v:.6f}")
    arr = np.array(final_val_mses)
    print(f"  Mean: {arr.mean():.6f}  Std: {arr.std():.6f}")


if __name__ == "__main__":
    main()
