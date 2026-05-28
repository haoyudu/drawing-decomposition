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
    # AuxK loss (Gao et al. 2024) to prevent dead features
    parser.add_argument("--auxk", type=int, default=None,
                        help="AuxK: number of dead features to activate on residual per step. "
                             "Recommended: k//2. Set to 0 to disable.")
    parser.add_argument("--auxk_coef", type=float, default=1/32,
                        help="Weight on the AuxK auxiliary loss (default: 1/32 per the paper)")
    parser.add_argument("--dead_steps_threshold", type=int, default=None,
                        help="Steps without activation before a feature is considered dead "
                             "(default: 2 * steps_per_epoch)")
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


def auxk_loss(model, batch, recons, dead_mask, k_aux):
    """
    AuxK auxiliary loss (Gao et al. 2024, §3.3).

    Among features that haven't activated recently (dead_mask), activate the
    top-k_aux on the reconstruction residual and compute MSE on that residual.
    This gives dead features gradient without interfering with the main loss.

    dead_mask: bool tensor of shape (M,), True = dead feature
    k_aux: how many dead features to activate per sample
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return torch.tensor(0.0, device=batch.device)

    k_aux = min(k_aux, n_dead)

    residual = (batch - recons).detach()  # shape (B, D)

    # Encode residual using only dead features
    # encode_pre_act gives (B, M); we'll zero out live features
    with torch.no_grad():
        x_pre, _ = model.preprocess(batch)
    pre_act = model.encode_pre_act(x_pre)           # (B, M)
    pre_act_dead = pre_act[:, dead_mask]             # (B, n_dead)

    # TopK among dead features only
    topk = torch.topk(pre_act_dead, k=k_aux, dim=-1)
    values = torch.relu(topk.values)
    aux_latents = torch.zeros_like(pre_act_dead)
    aux_latents.scatter_(-1, topk.indices, values)  # (B, n_dead)

    # Decode using dead decoder columns only
    dead_decoder = model.decoder.weight[:, dead_mask]  # (D, n_dead)
    aux_recons = aux_latents @ dead_decoder.t()         # (B, D)

    return ((aux_recons - residual) ** 2).mean()


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
    auxk = args.auxk if args.auxk is not None else args.k // 2
    use_auxk = auxk > 0
    print(f"\n=== Fold {fold_idx} | Model: Autoencoder(M={args.M}, D={D}, k={args.k}) ===")
    print(f"    Parameters: {n_params:,}  |  AuxK={'disabled' if not use_auxk else f'{auxk} (coef={args.auxk_coef})'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    steps_per_epoch = len(loader)
    dead_threshold = args.dead_steps_threshold if args.dead_steps_threshold is not None \
        else 2 * steps_per_epoch

    trajectory = []
    global_step = 0
    # steps_since_activated tracks how many steps each feature has gone without activating
    steps_since_activated = torch.zeros(args.M, dtype=torch.long, device=device)

    for epoch in tqdm(range(1, args.num_epochs + 1), desc=f"Fold {fold_idx} epochs"):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            latents_pre_act, latents, recons = model(batch)

            main_loss = ((recons - batch) ** 2).mean()

            if use_auxk:
                dead_mask = steps_since_activated >= dead_threshold   # (M,)
                aux = auxk_loss(model, batch, recons, dead_mask, auxk)
                loss = main_loss + args.auxk_coef * aux
            else:
                loss = main_loss

            loss.backward()
            optimizer.step()

            # renormalize decoder columns
            model.decoder.weight.data /= (
                model.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
            )

            # update dead-feature tracker
            activated = (latents > 0).any(dim=0)   # (M,) bool
            steps_since_activated[activated] = 0
            steps_since_activated[~activated] += 1

            train_loss_sum += main_loss.item() * len(batch)
            train_n += len(batch)
            global_step += 1

        train_mse = train_loss_sum / train_n
        v_mse = val_mse(model, val_tensor, args.batch_size, device)
        n_dead = (steps_since_activated >= dead_threshold).sum().item()
        print(f"  epoch={epoch:4d}  train_mse={train_mse:.6f}  val_mse={v_mse:.6f}"
              f"  dead={n_dead}/{args.M}")
        trajectory.append({"epoch": epoch, "train_mse": train_mse, "val_mse": v_mse,
                           "n_dead": n_dead})

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
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse", "n_dead"])
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
