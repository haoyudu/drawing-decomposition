# drawing-decomposition

Trains a TopK sparse autoencoder (SAE) on CLIP embeddings using k-fold cross-validation.

## Files

- `model.py` — verbatim copy of the OpenAI sparse autoencoder model ([source](https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py))
- `train_sae.py` — training script

## Environment

Run inside the `vislearnlab` conda environment on the lab server (provides PyTorch with CUDA, NumPy, pandas, scikit-learn, tqdm). Training will automatically use the GPU if available.

## Usage

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--embeddings_path` | *(required)* | Path to `.npy` file of shape `(N, D)` |
| `--n_folds` | `5` | Number of CV folds; `1` = single 90/10 split |
| `--k` | `32` | TopK sparsity (active features per sample) |
| `--M` | `4096` | Number of learned SAE features |
| `--lr` | `1e-4` | Adam learning rate |
| `--batch_size` | `512` | Training batch size |
| `--num_epochs` | `200` | Epochs per fold |
| `--seed` | `0` | Random seed for folds and shuffling |
| `--output_dir` | `./outputs` | Directory for saved models and loss CSVs |

### Sanity check (1 fold, 50 epochs)

```bash
conda activate vislearnlab
python train_sae.py --embeddings_path data/kiddraw_clip_embeddings.npy --n_folds 1 --num_epochs 50
```

### Full 5-fold run (all defaults)

```bash
conda activate vislearnlab
python train_sae.py --embeddings_path data/kiddraw_clip_embeddings.npy
```

## Outputs

For each fold `i`, the script writes:

- `outputs/fold{i}_losses.csv` — epoch-by-epoch `train_mse` and `val_mse`
- `outputs/fold{i}_sae.pt` — final model state dict

At the end, per-fold final validation MSE plus mean ± std are printed to stdout.
