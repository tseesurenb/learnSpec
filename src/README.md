# LearnSpec

**Learnable Spectral Collaborative Filtering**

LearnSpec learns spectral filters (polynomial or per-eigenvalue) on precomputed eigendecompositions of user-user and item-item similarity matrices to improve collaborative filtering recommendations.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, pandas, scikit-learn

```bash
pip install torch numpy scipy pandas scikit-learn

for h200:

pip install --user --force-reinstall pandas scikit-learn cupy-cuda12x

```

## Quick Start

```bash
cd src

# Step 1: Precompute eigendecompositions (one-time per dataset)
python precompute_eigen.py --dataset ml-100k --n_eigen 500 --size both

# Step 2: Evaluate baseline (no training)
python main.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --infer

# Step 3: Train
python main.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --epochs 200
```

## Best Commands Per Dataset

```bash
# ml-100k — NDCG@20: 0.4602 (baseline, no training needed)
python main.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --f_poly bernstein --f_order 24 --f_init bandpass --infer

# gowalla — NDCG@20: 0.1514 (baseline, beta=0.4)
 python main.py --dataset gowalla --u_eigen 310 --i_eigen 2000 --beta 0.4 --f_order 24 --f_init bandpass --decay 1e-02 --lr 0.001

# lastfm — NDCG@20: 0.2155
python main.py --dataset lastfm --u_eigen 500 --i_eigen 50 --f_poly bernstein --f_order 16 --lr 0.01 --epochs 400 --patience 50

# yelp2018 — NDCG@20: 0.0619, RECALL@20: 0.0743 (baseline, beta=0.4)
python main.py --dataset yelp2018 --u_eigen 290 --i_eigen 1900 --beta 0.4 --infer --f_init bandpass

# amazon-book — NDCG@20: 0.0613, RECALL@20: 0.0759 (baseline, beta=0.25)
python main.py --dataset amazon-book --u_eigen 2000 --i_eigen 14000 --beta 0.25 --infer --f_init lowpass

```

## Project Structure

```
src/
├── main.py              # Training and evaluation entry point
├── model.py             # LearnSpecCF model
├── filter.py            # Spectral filters (APSFilter, DirectFilter, AdaptiveFilter)
├── procedure.py         # Training loop (MSE) and evaluation (NDCG, Recall)
├── config.py            # CLI args and configuration
├── utils.py             # Utilities (metrics, data splits, I/O)
├── dataloader.py        # Dataset loaders (ML-100K, LastFM, Gowalla, Yelp, Amazon)
├── precompute_eigen.py  # Eigendecomposition via randomized SVD
└── search.py            # Grid search for baseline hyperparameters
```

## Method

1. **Precompute**: Eigendecomposition of normalized similarity matrices via randomized SVD
2. **Filter**: Learn spectral filter h(λ) applied to eigenvalues (sigmoid-bounded, 0-1)
3. **Predict**: Reconstruct ratings as `U · diag(h(λ)) · U^T · R` for user view (analogous for item view)
4. **Train**: MSE loss on held-out validation interactions (70/30 split)
5. **Evaluate**: Transfer best filter to full-data model, measure NDCG@20 and Recall@20 on test set

## CLI Reference

### `main.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | ml-100k | ml-100k, lastfm, gowalla, yelp2018, amazon-book |
| `--view` | ui | Spectral view: u, i, or ui (both) |
| `--u_eigen` | 25 | Number of user eigenvalues |
| `--i_eigen` | 130 | Number of item eigenvalues |
| `--beta` | 0.5 | Degree normalization exponent |
| `--f_poly` | bernstein | Filter type: bernstein, cheby, direct, adaptive |
| `--f_order` | 32 | Polynomial order (K+1 coefficients) |
| `--f_init` | bandpass | Init shape: uniform, lowpass, highpass, bandpass |
| `--f_act` | sigmoid | Activation: sigmoid, softplus, tanh, none |
| `--opt` | rmsprop | Optimizer: rmsprop, adam |
| `--lr` | 0.001 | Learning rate |
| `--decay` | 0 | Weight decay |
| `--epochs` | 200 | Max training epochs |
| `--patience` | 5 | Early stopping patience |
| `--eval_every` | 5 | Evaluate every N epochs |
| `--infer` | false | Baseline only, no training |

### `precompute_eigen.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Dataset name |
| `--n_eigen` | (required) | Number of eigenvalues to compute |
| `--size` | both | full, partial, or both |
| `--beta` | 0.5 | Degree normalization |
| `--n_iter` | 10 | SVD iterations (higher = more accurate) |

### `search.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | ml-100k | Dataset name |
| `--eigen_step` | 50 | Grid step size for eigenvalue search |

## Filter Types

| Type | `--f_poly` | Params | Description |
|------|-----------|--------|-------------|
| Polynomial (Bernstein) | `bernstein` | K+1 | Smooth spectral filter on [0,1] |
| Polynomial (Chebyshev) | `cheby` | K+1 | Spectral filter on [-1,1] |
| Direct | `direct` | n_eigen | One free parameter per eigenvalue |
| Adaptive | `adaptive` | K+1 + n_eigen | Polynomial + per-eigenvalue corrections |

## Data Format

Datasets are stored in `../data/{name}/` with:
- `train.txt`: User-item interactions (whitespace-separated, first column is user ID)
- `test.txt`: Same format for test split

## Cache

Precomputed eigendecompositions are saved as compressed `.npz` files in `../cache/{dataset}/`. File naming encodes the parameters (view, n_eigen, beta, split seed/ratio) so different configurations coexist.

## Citation

```
@article{batsuuri2025learnspec,
  title={LearnSpec: Learnable Spectral Collaborative Filtering},
  author={Batsuuri, Tseesuren},
  year={2025}
}
```
