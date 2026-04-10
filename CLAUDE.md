# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LearnSpec** — Learnable Spectral Collaborative Filtering. A PyTorch-based recommender system research project by Tseesuren Batsuuri (Macquarie University).

The core idea: learn spectral filters (polynomial or per-eigenvalue) on precomputed eigendecompositions of user-user and item-item similarity matrices to improve collaborative filtering recommendations. Uses sigmoid activation for bounded (0-1) filter responses.

## Running the Code

All commands run from `src/`:

```bash
# Basic training (ml-100k, default settings)
python main.py --dataset ml-100k --epochs 100

# Precompute eigendecompositions (required before first run on a dataset)
python gen_eigen.py --dataset gowalla --view u --n_eigen 1200 --size both
python gen_eigen.py --dataset gowalla --view i --n_eigen 300 --size both

# Baseline-only search (no training, finds best beta and eigenvalue counts)
python search.py --dataset ml-100k --eigen_step 50
```

### Best Known Commands

```bash
# ml-100k (NDCG@20: 0.4602, baseline — no training needed)
python main.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --f_poly bernstein --f_order 24 --f_init bandpass --infer

# gowalla (NDCG@20: 0.1514, baseline, beta=0.4)
python main.py --dataset gowalla --u_eigen 200 --i_eigen 700 --beta 0.4 --infer

# lastfm (NDCG@20: 0.2155)
python main.py --dataset lastfm --u_eigen 500 --i_eigen 50 --f_poly bernstein --f_order 16 --lr 0.01 --epochs 400 --patience 50

# yelp2018 (NDCG@20: 0.0617, baseline, beta=0.4)
python main.py --dataset yelp2018 --u_eigen 300 --i_eigen 2000 --beta 0.4 --infer
```

**Note:** Degree normalization uses symmetric formula: `A_n = D_u^{-beta} R D_i^{-beta}`. Both `gen_eigen.py` and `fast_gen_eigen.py` use this formula. Optimal beta varies by dataset (ml-100k: 0.5, gowalla: 0.4, yelp2018: 0.4). Use `--u_beta`/`--i_beta` to set per-view beta if needed.

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | ml-100k | Dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book |
| `--sim` | ui | Spectral view: u (user), i (item), ui (both) |
| `--u_eigen` | 100 | Number of user eigenvalues |
| `--i_eigen` | 300 | Number of item eigenvalues |
| `--beta` | 0.5 | Degree normalization for both views |
| `--u_beta` | None | Override beta for user view only |
| `--i_beta` | None | Override beta for item view only |
| `--f_poly` | bernstein | Filter type: bernstein, cheby |
| `--f_order` | 32 | Polynomial order K (K+1 learnable coefficients) |
| `--f_init` | uniform | Initial filter shape: uniform, lowpass, highpass, bandpass |
| `--f_drop` | 0.0 | Spectral dropout: probability of masking eigencomponents |
| `--optimizer` | rmsprop | Optimizer: rmsprop, adam |
| `--lr` | 0.001 | Learning rate |
| `--decay` | 0 | Weight decay (L2 regularization) |
| `--epochs` | 5 | Maximum training epochs |
| `--patience` | 5 | Early stopping patience (in eval steps) |
| `--eval_every` | 5 | Evaluate validation every N epochs |
| `--batch_size` | 1024 | Batch size for train/test |

## Architecture

### Data Flow

1. **Eigendecomposition** (`gen_eigen.py`): Normalizes adjacency matrix `A_n = D_u^{-beta} R D_i^{-beta}`, then computes eigendecomposition of user similarity `S_u = A_n @ A_n^T` and item similarity `S_i = A_n^T @ A_n`. Each view can use a different beta. Cached in `../cache/{dataset}/`. Fast alternative: `fast_gen_eigen.py` computes both views via single SVD of `A_n` (same beta for both views).
2. **Model init** (`model.py` -> `LearnSpecCF`): Loads cached eigen data, creates `LearnableSpectralFilter` for active views (user/item/both based on `--sim`).
3. **Filter** (`filters.py`): Single polynomial filter with sigmoid activation. Filter types:
   - `bernstein`: Bernstein polynomial basis on [0,1]
   - `cheby`: Chebyshev polynomial basis on [-1,1]
4. **Training** (`procedure.py` -> `MSE_train_spectral`): MSE loss on validation interactions. Training data split 70/30 (train/validation), learns filter parameters, then evaluates on test with full data.
5. **Evaluation** (`procedure.py` -> `Test`, `Test_val`): Computes NDCG@20, Recall@20, Precision@20.

### Key Design Patterns

- **Train/val/test split**: Training data is split 70/30. Model is built on partial adjacency matrix (70%), trained to predict held-out validation interactions (30%), then best parameters are transferred to a full-data model for final test evaluation.
- **Eigenvalue caching**: Eigen decompositions are expensive. Cached to `../cache/{dataset}/` with filenames encoding view, count, beta, and split parameters. Run `gen_eigen.py` before first use of a dataset.
- **Filter parameter saving**: Best filter parameters saved as pickle files to `../results/filter_params/`.
- **Run logging**: Every training run saves a JSON summary to `../runs/train/{dataset}/`.

### Datasets

Stored in `../data/{name}/` with `train.txt` and `test.txt` (whitespace-separated user-item pairs). Supported: ml-100k, lastfm, gowalla, yelp2018, amazon-book, last-fm-big.

## Dependencies

PyTorch, NumPy, SciPy, pandas, scikit-learn. GPU support via `cupy-cuda12x` (optional).
