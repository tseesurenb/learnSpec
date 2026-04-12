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
# ml-100k (NDCG@20: 0.4647, trained, bernstein+BPR+MSE)
python main.py --dataset ml-100k --u_eigen 20 --i_eigen 50 --beta 0.3 --f_poly bernstein --f_order 32 --f_init lowpass --lr 0.01 --decay 0.1 --epochs 300 --mse_weight 0.5

# gowalla (NDCG@20: 0.1584, baseline, bernstein+rise+softplus)
python main.py --dataset gowalla --u_eigen 310 --i_eigen 2000 --beta 0.4 --f_poly bernstein --f_order 4 --f_init rise --f_act softplus --infer

# gowalla (NDCG@20: 0.1575, direct filter baseline)
python main.py --dataset gowalla --u_eigen 320 --i_eigen 2500 --beta 0.4 --f_poly direct --f_init lowpass --infer

# gowalla (NDCG@20: 0.1572, direct filter trained from highpass, +26.9%)
python main.py --dataset gowalla --u_eigen 320 --i_eigen 2500 --beta 0.4 --f_poly direct --f_init highpass --opt adam --lr 0.5 --decay 0.01

# yelp2018 (NDCG@20: 0.0622, direct filter baseline/trained)
python main.py --dataset yelp2018 --u_eigen 300 --i_eigen 1900 --beta 0.4 --f_poly direct --f_init butterworth --infer
# or trained from highpass:
python main.py --dataset yelp2018 --u_eigen 300 --i_eigen 1900 --beta 0.4 --f_poly direct --f_init highpass --opt adam --lr 0.5 --decay 0.05

# amazon-book (NDCG@20: 0.0621, Recall: 0.0761, baseline)
python main.py --dataset amazon-book --u_eigen 2000 --i_eigen 14000 --beta 0.25 --infer

# lastfm (NDCG@20: 0.2155)
python main.py --dataset lastfm --u_eigen 500 --i_eigen 50 --f_poly bernstein --f_order 16 --lr 0.01 --epochs 400 --patience 50
```

**Note:** Degree normalization uses symmetric formula: `A_n = D_u^{-beta} R D_i^{-beta}`. Both `gen_eigen.py` and `fast_gen_eigen.py` use this formula. Optimal beta varies by dataset (ml-100k: 0.3, gowalla: 0.4, yelp2018: 0.4, amazon-book: 0.25). Use `--u_beta`/`--i_beta` to set per-view beta if needed.

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | ml-100k | Dataset: ml-100k, lastfm, gowalla, yelp2018, amazon-book |
| `--view` | ui | Spectral view: u (user), i (item), ui (both) |
| `--u_eigen` | 20 | Number of user eigenvalues |
| `--i_eigen` | 50 | Number of item eigenvalues |
| `--beta` | 0.4 | Degree normalization for both views |
| `--f_poly` | direct | Filter type: bernstein, cheby, direct |
| `--f_order` | 32 | Polynomial order K (for bernstein/cheby only) |
| `--f_init` | lowpass | Init shape: uniform, lowpass, highpass, bandpass, butterworth, decay, rise |
| `--f_act` | sigmoid | Activation: sigmoid, softplus |
| `--f_drop` | 0.0 | Spectral dropout: probability of masking eigencomponents |
| `--f_reg` | 0.0 | Smoothness regularization weight |
| `--loss` | bpr | Loss function: bpr, mse |
| `--opt` | adam | Optimizer: rmsprop, adam |
| `--lr` | 0.5 | Learning rate |
| `--decay` | 0.01 | Weight decay (L2 regularization) |
| `--epochs` | 50 | Maximum training epochs |
| `--patience` | 10 | Early stopping patience (in eval steps) |
| `--eval_every` | 5 | Evaluate validation every N epochs |
| `--batch_size` | 1024 | Batch size for train/test |
| `--infer` | false | Baseline only, no training |

## Architecture

### Data Flow

1. **Eigendecomposition** (`gen_eigen.py`): Normalizes adjacency matrix `A_n = D_u^{-beta} R D_i^{-beta}`, then computes eigendecomposition of user similarity `S_u = A_n @ A_n^T` and item similarity `S_i = A_n^T @ A_n`. Each view can use a different beta. Cached in `../cache/{dataset}/`. Fast alternative: `fast_gen_eigen.py` computes both views via single SVD of `A_n` (same beta for both views).
2. **Model init** (`model.py` -> `LearnSpecCF`): Loads cached eigen data, creates `LearnableSpectralFilter` for active views (user/item/both based on `--sim`).
3. **Filter** (`filter.py`): Spectral filter with activation (sigmoid or softplus). Filter types:
   - `bernstein`: Bernstein polynomial basis on [0,1] (K+1 params)
   - `cheby`: Chebyshev polynomial basis on [-1,1] (K+1 params)
   - `direct`: One learnable parameter per eigenvalue (n_eigen params). Acts as spectral filter oracle — discovers the optimal per-eigenvalue response from data.
4. **Training** (`procedure.py` -> `train_spectral`): BPR or MSE loss on validation interactions. Training data split 70/30 (train/validation), learns filter parameters, then evaluates on test with full data.
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
