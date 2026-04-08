import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd

from dataloader import Dataset

DATASETS = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
SPLIT_RATIO = 0.7
SPLIT_SEED = 42


def split_training_data(dataset, split_ratio=0.70, seed=42):
    np.random.seed(seed)
    partial_train, validation = {}, {}
    for uid in range(dataset.n_users):
        items = list(dataset.allPos[uid])
        if len(items) > 1:
            shuffled = items.copy()
            np.random.shuffle(shuffled)
            sp = max(1, int(len(shuffled) * split_ratio))
            partial_train[uid] = shuffled[:sp]
            validation[uid] = shuffled[sp:]
        else:
            partial_train[uid] = items
            validation[uid] = []
    return partial_train, validation


def create_partial_adj_matrix(partial_train, n_users, n_items):
    from scipy.sparse import csr_matrix as sp_csr
    rows, cols = [], []
    for uid, items in partial_train.items():
        for iid in items:
            rows.append(uid)
            cols.append(iid)
    return sp_csr(([1] * len(rows), (rows, cols)), shape=(n_users, n_items))


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute eigendecomposition via randomized SVD')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS)
    parser.add_argument('--n_eigen', type=int, required=True)
    parser.add_argument('--size', type=str, default='both', choices=['full', 'partial', 'both'])
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--split_ratio', type=float, default=SPLIT_RATIO)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--eigen_dir', type=str, default='../cache')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n_iter', type=int, default=10)
    return parser.parse_args()


def load_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    dataset = Dataset(path=f"../data/{dataset_name}")
    print(f"Loaded: {dataset.n_users} users, {dataset.m_items} items, {dataset.UserItemNet.nnz} interactions")
    return dataset


def compute_normalized_adj(R, beta=0.5):
    from scipy.sparse import diags
    d_u = np.array(R.sum(axis=1)).flatten().astype(np.float64)
    d_i = np.array(R.sum(axis=0)).flatten().astype(np.float64)
    d_u_inv = np.where(d_u > 0, d_u ** (-beta), 0)
    d_i_inv = np.where(d_i > 0, d_i ** (-beta), 0)
    return diags(d_u_inv) @ R.astype(np.float64) @ diags(d_i_inv)


def fast_eigen_svd(A_n, n_eigen, n_iter=10, seed=42):
    N, M = A_n.shape
    k = min(n_eigen, min(N, M) - 1)

    # Try GPU (CuPy) for sparse matmuls, CPU for final dense SVD
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as cp_csr

        print(f"Computing randomized SVD (GPU+CPU): A_n ({N}x{M}), k={k}, n_iter={n_iter}")
        t0 = time.time()

        A_gpu = cp_csr(A_n.astype(np.float32))
        At_gpu = cp_csr(A_n.T.tocsr().astype(np.float32))

        # Random projection on GPU
        rng = cp.random.RandomState(seed)
        Omega = rng.standard_normal((M, k + 10), dtype=cp.float32)
        Y = A_gpu @ Omega
        Q, _ = cp.linalg.qr(Y)
        del Omega, Y

        # Power iteration on GPU (sparse matmuls)
        for i in range(n_iter):
            Z = At_gpu @ Q
            Q2, _ = cp.linalg.qr(Z)
            Y = A_gpu @ Q2
            Q, _ = cp.linalg.qr(Y)
            del Z, Y
            print(f"    Power iteration {i+1}/{n_iter} ({time.time()-t0:.1f}s)")

        # Two-sided projection: B = Q^T @ A @ A^T @ Q = small (k+10, k+10) matrix
        # Then eigendecompose B to get singular values and left vectors
        # Right vectors recovered via V = A^T @ Q @ Uhat @ diag(1/sigma)
        AQ = A_gpu @ (At_gpu @ Q)  # reuse: A @ A^T @ Q on GPU
        B = cp.asnumpy(Q.T @ AQ)   # (k+10, k+10) — tiny
        Q_cpu = cp.asnumpy(Q)
        AtQ_cpu = cp.asnumpy(At_gpu @ Q)  # (M, k+10) for recovering V
        del A_gpu, At_gpu, Q, Q2, AQ
        cp.get_default_memory_pool().free_all_blocks()

        print(f"    Computing dense eigen on CPU: B {B.shape} ({time.time()-t0:.1f}s)")
        # B is symmetric positive semi-definite, eigenvalues = sigma^2
        eigenvals_B, Uhat = np.linalg.eigh(B)
        # eigh returns ascending order, flip to descending
        idx = np.argsort(eigenvals_B)[::-1]
        eigenvals_B = eigenvals_B[idx]
        Uhat = Uhat[:, idx]

        sigma = np.sqrt(np.maximum(eigenvals_B[:k], 0))
        U = Q_cpu @ Uhat[:, :k]
        # V = A^T @ Q @ Uhat @ diag(1/sigma)
        Vt = (AtQ_cpu @ Uhat[:, :k] @ np.diag(1.0 / (sigma + 1e-10))).T

        print(f"  Completed in {time.time() - t0:.1f}s")

    except (ImportError, Exception) as e:
        print(f"GPU not available ({e}), using CPU")
        print(f"Computing randomized SVD: A_n ({N}x{M}), k={k}, n_iter={n_iter}")
        t0 = time.time()
        U, sigma, Vt = randomized_svd(A_n, n_components=k, n_iter=n_iter, random_state=seed)
        print(f"  CPU SVD completed in {time.time() - t0:.1f}s")

    eigenvals = sigma ** 2
    print(f"  Eigenvalues range: [{eigenvals[-1]:.6f}, {eigenvals[0]:.6f}]")
    print(f"  User eigenvectors: {U.shape}, Item eigenvectors: {Vt.T.shape}")
    return eigenvals, U, eigenvals.copy(), Vt.T


def save_eigen(filepath, dataset_name, view, eigenvals, eigenvecs, beta, extra_meta=None):
    save_kwargs = dict(
        eigenvals=eigenvals.astype(np.float32),
        eigenvecs=eigenvecs.astype(np.float32),
        dataset=np.array(dataset_name), view=np.array(view),
        which=np.array('LM'), n_eigen=np.array(len(eigenvals)),
        u_beta=np.array(beta, dtype=np.float32),
        i_beta=np.array(beta, dtype=np.float32),
        method=np.array('randomized_svd'),
    )
    if extra_meta:
        for k, v in extra_meta.items():
            save_kwargs[k] = np.array(v)

    np.savez_compressed(filepath, **save_kwargs)
    actual_path = filepath if filepath.endswith('.npz') else filepath + '.npz'
    print(f"  Saved: {os.path.basename(actual_path)} ({os.path.getsize(actual_path) / (1024*1024):.1f} MB)")


def format_beta_string(beta):
    return str(beta).replace('.', 'p')


def generate(args, R, dataset_name, cache_dir, prefix, extra_meta=None):
    A_n = compute_normalized_adj(R, beta=args.beta)
    beta_str = format_beta_string(args.beta)
    eigenvals_u, eigenvecs_u, eigenvals_i, eigenvecs_i = fast_eigen_svd(
        A_n, args.n_eigen, n_iter=args.n_iter, seed=args.seed)

    for view, evals, evecs in [('user', eigenvals_u, eigenvecs_u), ('item', eigenvals_i, eigenvecs_i)]:
        filename = f"{prefix}{dataset_name}_{view}_largestEigen_n{len(evals)}_degNorm_{beta_str}.npz"
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath) and not args.overwrite:
            print(f"  Exists (skip): {filename}. Use --overwrite to replace.")
        else:
            v = 'u' if view == 'user' else 'i'
            save_eigen(filepath, dataset_name, v, evals, evecs, args.beta, extra_meta)


def main():
    args = parse_args()
    cache_dir = os.path.join(os.path.abspath(args.eigen_dir), args.dataset)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Precompute Eigendecomposition (Randomized SVD)")
    print(f"Dataset: {args.dataset}, n_eigen: {args.n_eigen}, beta: {args.beta}")
    print(f"Size: {args.size}, n_iter: {args.n_iter}")
    print(f"Output: {cache_dir}")
    print(f"{'='*60}")

    dataset = load_dataset(args.dataset)

    if args.size in ['full', 'both']:
        print(f"\n--- Full eigendecomposition ---")
        generate(args, dataset.UserItemNet, args.dataset, cache_dir, prefix='full_')

    if args.size in ['partial', 'both']:
        print(f"\n--- Partial eigendecomposition (ratio={args.split_ratio}, seed={args.seed}) ---")
        partial_train, _ = split_training_data(dataset, split_ratio=args.split_ratio, seed=args.seed)
        R_partial = create_partial_adj_matrix(partial_train, dataset.n_users, dataset.m_items)

        beta_str = format_beta_string(args.beta)
        ratio_int = int(args.split_ratio * 100)
        extra_meta = {'split_ratio': args.split_ratio, 'split_seed': args.seed}

        A_n = compute_normalized_adj(R_partial, beta=args.beta)
        eigenvals_u, eigenvecs_u, eigenvals_i, eigenvecs_i = fast_eigen_svd(
            A_n, args.n_eigen, n_iter=args.n_iter, seed=args.seed)

        for view, evals, evecs in [('user', eigenvals_u, eigenvecs_u), ('item', eigenvals_i, eigenvecs_i)]:
            filename = f"partial_{args.dataset}_{view}_largestEigen_n{len(evals)}_degNorm_{beta_str}_seed_{args.seed}_ratio_{ratio_int}.npz"
            filepath = os.path.join(cache_dir, filename)
            if os.path.exists(filepath) and not args.overwrite:
                print(f"  Exists (skip): {filename}")
            else:
                v = 'u' if view == 'user' else 'i'
                save_eigen(filepath, args.dataset, v, evals, evecs, args.beta, extra_meta)

    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == '__main__':
    main()
