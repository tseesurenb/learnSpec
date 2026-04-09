"""
Eigen search: find optimal beta x u_eigen x i_eigen for each filter configuration.

Auto-generates eigendecompositions for missing betas.

Usage:
    # Fast mode: uniform filter only, find best beta x u_eigen x i_eigen
    python eigen_search.py --dataset ml-100k --eigen_step 10 --fast

    # Full mode: search all filter configs (bernstein/cheby x 5 inits)
    python eigen_search.py --dataset ml-100k --eigen_step 10

    # Search specific filter only
    python eigen_search.py --dataset ml-100k --eigen_step 25 --f_poly bernstein --f_init bandpass

    # Search with specific betas (auto-generates if missing)
    python eigen_search.py --dataset gowalla --betas 0.3 0.4 0.5 0.6 --n_eigen 1200 --fast

    # Limit search range
    python eigen_search.py --dataset gowalla --eigen_step 100 --max_u 1200 --max_i 700
"""

import argparse
import os
import sys
import csv
import torch
import numpy as np
import gc
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import Dataset
from model import LearnSpecCF
from procedure import evaluate
from filter import create_filter
from precompute_eigen import compute_normalized_adj, fast_eigen_svd, save_eigen

POLYS = ['bernstein', 'cheby']
INITS = ['uniform', 'lowpass', 'highpass', 'bandpass', 'butterworth']
F_ORDER = 24
F_ACT = 'sigmoid'


def parse_args():
    parser = argparse.ArgumentParser(description='Eigen search per filter config')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--eigen_step', type=int, default=25, help='Step size for eigen grid')
    parser.add_argument('--max_u', type=int, default=None, help='Max user eigenvalues to search')
    parser.add_argument('--max_i', type=int, default=None, help='Max item eigenvalues to search')
    parser.add_argument('--n_eigen', type=int, default=500, help='Number of eigenvalues to generate if missing')
    parser.add_argument('--betas', type=float, nargs='+', default=None,
                        help='Beta values to search (auto-generates if missing). Default: use available from cache')
    parser.add_argument('--f_poly', type=str, default=None, choices=POLYS, help='Search only this poly type')
    parser.add_argument('--f_init', type=str, default=None, choices=INITS, help='Search only this init type')
    parser.add_argument('--fast', action='store_true', help='Fast mode: uniform filter only, 1 config instead of 10')
    parser.add_argument('--n_iter', type=int, default=10, help='Power iterations for randomized SVD')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def get_available_betas(cache_dir, dataset_name):
    betas = set()
    if not os.path.exists(cache_dir):
        return []
    for f in os.listdir(cache_dir):
        if f.startswith(f"full_{dataset_name}_") and 'degNorm_' in f and (f.endswith('.pkl') or f.endswith('.npz')):
            try:
                beta_str = f.split('degNorm_')[1].split('_')[0].split('.')[0]
                betas.add(float(beta_str.replace('p', '.')))
            except (ValueError, IndexError):
                continue
    return sorted(betas)


def get_max_eigen(cache_dir, dataset_name, view, beta):
    beta_str = str(beta).replace('.', 'p')
    view_name = 'user' if view == 'u' else 'item'
    max_n = 0
    if not os.path.exists(cache_dir):
        return 0
    for f in os.listdir(cache_dir):
        pattern = f"full_{dataset_name}_{view_name}_largestEigen_n"
        for ext in ('.npz', '.pkl'):
            if f.startswith(pattern) and f.endswith(f"_degNorm_{beta_str}{ext}"):
                try:
                    n_str = f[len(pattern):f.index('_degNorm_')]
                    max_n = max(max_n, int(n_str))
                except (ValueError, IndexError):
                    continue
    return max_n


def has_eigen_files(cache_dir, dataset_name, beta):
    """Check if both user and item eigen files exist for this beta."""
    max_u = get_max_eigen(cache_dir, dataset_name, 'u', beta)
    max_i = get_max_eigen(cache_dir, dataset_name, 'i', beta)
    return max_u > 0 and max_i > 0


def generate_eigen(dataset_name, beta, n_eigen, cache_dir, n_iter=10):
    """Generate eigendecomposition for a given beta if not already cached."""
    if has_eigen_files(cache_dir, dataset_name, beta):
        max_u = get_max_eigen(cache_dir, dataset_name, 'u', beta)
        max_i = get_max_eigen(cache_dir, dataset_name, 'i', beta)
        print(f"  Beta={beta}: eigen files exist (u={max_u}, i={max_i})")
        return True

    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n  Generating eigendecomposition: beta={beta}, n_eigen={n_eigen}")

    dataset = Dataset(path=f"../data/{dataset_name}")
    R = dataset.UserItemNet

    t0 = time.time()
    A_n = compute_normalized_adj(R, beta=beta)
    eigenvals_u, eigenvecs_u, eigenvals_i, eigenvecs_i = fast_eigen_svd(
        A_n, n_eigen, n_iter=n_iter, seed=42)

    beta_str = str(beta).replace('.', 'p')
    for view, evals, evecs in [('user', eigenvals_u, eigenvecs_u), ('item', eigenvals_i, eigenvecs_i)]:
        filename = f"full_{dataset_name}_{view}_largestEigen_n{len(evals)}_degNorm_{beta_str}.npz"
        filepath = os.path.join(cache_dir, filename)
        v = 'u' if view == 'user' else 'i'
        save_eigen(filepath, dataset_name, v, evals, evecs, beta)

    del dataset
    gc.collect()
    print(f"  Generated in {time.time() - t0:.1f}s")
    return True


def make_config(dataset_name, u_eigen, i_eigen, beta, poly, f_init, device):
    return {
        'dataset': dataset_name, 'seed': 42, 'view': 'ui',
        'u_eigen': u_eigen, 'i_eigen': i_eigen, 'beta': beta,
        'f_order': F_ORDER, 'f_init': f_init, 'poly': poly,
        'f_dropout': 0.0, 'f_act': F_ACT,
        'opt': 'rmsprop', 'lr': 0.01, 'decay': 0.1,
        'loss': 'bpr',
        'epochs': 0, 'batch_size': 1024, 'patience': 10,
        'eval_every': 5, 'split_ratio': 0.7,
        'infer': True, 'save': False, 'log': False,
        'device': device, 'topks': [20],
    }


def eval_config(dataset, config, device):
    try:
        model = LearnSpecCF(dataset.UserItemNet, config, use_cache=True, verbose=False).to(device)
        model.eval()
        with torch.no_grad():
            results = evaluate(dataset, model, split='test', batch_size=config['batch_size'])
        ndcg, recall = results['ndcg'][0], results['recall'][0]
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ndcg, recall
    except Exception as e:
        print(f"    eval error: {e}")
        return None


def search_one_filter(dataset_name, poly, f_init, betas, device, args, writer, f_out, done):
    """Search beta x u_eigen x i_eigen for one filter config."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', dataset_name)
    step = args.eigen_step

    filter_label = f"{poly}(K={F_ORDER}) | {f_init} | {F_ACT}"
    print(f"\n{'='*70}")
    print(f"  Filter: {filter_label}")
    print(f"{'='*70}")

    best_overall = {'ndcg': 0}

    for beta in betas:
        avail_max_u = get_max_eigen(cache_dir, dataset_name, 'u', beta)
        avail_max_i = get_max_eigen(cache_dir, dataset_name, 'i', beta)
        if args.max_u:
            avail_max_u = min(avail_max_u, args.max_u)
        if args.max_i:
            avail_max_i = min(avail_max_i, args.max_i)
        if avail_max_u == 0 or avail_max_i == 0:
            print(f"  Beta={beta}: no eigendecompositions found, skipping")
            continue

        u_values = list(range(step, avail_max_u + 1, step))
        i_values = list(range(step, avail_max_i + 1, step))
        if not u_values or not i_values:
            continue

        print(f"\n  Beta={beta} | u: {u_values[0]}..{u_values[-1]} | i: {i_values[0]}..{i_values[-1]} | {len(u_values)*len(i_values)} evals")
        print(f"  {'u':>6} {'i':>6} {'NDCG':>8} {'Recall':>8}")
        print(f"  {'-'*32}")

        dataset = Dataset(path=f"../data/{dataset_name}")
        best_beta_ndcg = 0
        u_no_improve, prev_u_best = 0, 0

        for u_e in u_values:
            row_best, i_no_improve = 0, 0
            for i_e in i_values:
                key = (poly, f_init, str(beta), str(u_e), str(i_e))
                if key in done:
                    continue

                cfg = make_config(dataset_name, u_e, i_e, beta, poly, f_init, device)
                result = eval_config(dataset, cfg, device)

                if result:
                    ndcg, recall = result
                    improved = ndcg > best_beta_ndcg
                    if improved:
                        best_beta_ndcg = ndcg
                    if ndcg > best_overall['ndcg']:
                        best_overall = {'beta': beta, 'u_eigen': u_e, 'i_eigen': i_e,
                                        'ndcg': ndcg, 'recall': recall}

                    row = {
                        'f_poly': poly, 'f_init': f_init, 'beta': beta,
                        'u_eigen': u_e, 'i_eigen': i_e,
                        'ndcg': f'{ndcg:.4f}', 'recall': f'{recall:.4f}',
                    }
                    writer.writerow(row)
                    f_out.flush()
                    done.add(key)

                    print(f"  {u_e:>6} {i_e:>6} {ndcg:>8.4f} {recall:>8.4f}{' *' if improved else ''}", flush=True)

                    if ndcg > row_best:
                        row_best, i_no_improve = ndcg, 0
                    else:
                        i_no_improve += 1
                        if i_no_improve >= 3:
                            break
                else:
                    break

            if row_best > prev_u_best:
                prev_u_best, u_no_improve = row_best, 0
            else:
                u_no_improve += 1
                if u_no_improve >= 3:
                    break

        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if best_beta_ndcg > 0:
            print(f"  >> Beta={beta} best NDCG={best_beta_ndcg:.4f}")

    if best_overall['ndcg'] > 0:
        print(f"\n  ** {filter_label} BEST: beta={best_overall['beta']}, "
              f"u={best_overall['u_eigen']}, i={best_overall['i_eigen']} => "
              f"NDCG={best_overall['ndcg']:.4f}, Recall={best_overall['recall']:.4f}")

    return best_overall


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using {'GPU: ' + torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', args.dataset)

    # Determine betas to search
    if args.betas:
        betas = sorted(args.betas)
    else:
        betas = get_available_betas(cache_dir, args.dataset)
        if not betas:
            betas = [0.5]
            print(f"  No cached eigendecompositions found. Will generate for beta={betas}")

    # Auto-generate missing eigen files
    for beta in betas:
        if not has_eigen_files(cache_dir, args.dataset, beta):
            print(f"  Beta={beta}: eigen files missing, generating (n_eigen={args.n_eigen})...")
            generate_eigen(args.dataset, beta, args.n_eigen, cache_dir, n_iter=args.n_iter)

    # Fast mode: single uniform filter (10x faster)
    if args.fast:
        polys = ['bernstein']
        inits = ['uniform']
    else:
        polys = [args.f_poly] if args.f_poly else POLYS
        inits = [args.f_init] if args.f_init else INITS

    total_filters = len(polys) * len(inits)
    mode_label = "FAST" if args.fast else "FULL"
    print(f"\nEigen Search ({mode_label}): {args.dataset}")
    print(f"  Filters: {total_filters} ({polys} x {inits})")
    print(f"  Betas: {betas}")
    print(f"  Eigen step: {args.eigen_step}")

    # Setup CSV
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    os.makedirs(results_dir, exist_ok=True)
    suffix = '_fast' if args.fast else ''
    csv_path = os.path.join(results_dir, f'eigen_search_{args.dataset}{suffix}.csv')
    fieldnames = ['f_poly', 'f_init', 'beta', 'u_eigen', 'i_eigen', 'ndcg', 'recall']

    # Resume
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['f_poly'], row['f_init'], row['beta'], row['u_eigen'], row['i_eigen'])
                done.add(key)
        print(f"  Resuming: {len(done)} evals already done")
        f_out = open(csv_path, 'a', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        f_out = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

    # Search each filter config
    summary = []
    for poly, f_init in product(polys, inits):
        best = search_one_filter(args.dataset, poly, f_init, betas, device, args, writer, f_out, done)
        if best['ndcg'] > 0:
            best['f_poly'] = poly
            best['f_init'] = f_init
            summary.append(best)

    f_out.close()

    # Print summary
    summary.sort(key=lambda x: x['ndcg'], reverse=True)
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: {args.dataset} -- Best eigen cuts per filter")
    print(f"{'='*70}")
    print(f"  {'Poly':>9} | {'Init':>11} | {'Beta':>5} | {'u_eigen':>7} | {'i_eigen':>7} | {'NDCG':>7} | {'Recall':>7}")
    print(f"  {'-'*66}")
    for i, r in enumerate(summary):
        best_marker = '  <-- BEST' if i == 0 else ''
        print(f"  {r['f_poly']:>9} | {r['f_init']:>11} | {r['beta']:>5.2f} | {r['u_eigen']:>7} | {r['i_eigen']:>7} | "
              f"{r['ndcg']:>7.4f} | {r['recall']:>7.4f}{best_marker}")

    if summary:
        b = summary[0]
        print(f"\n  Best command:")
        print(f"    python main.py --dataset {args.dataset} --f_poly {b['f_poly']} --f_init {b['f_init']} "
              f"--beta {b['beta']} --u_eigen {b['u_eigen']} --i_eigen {b['i_eigen']}")

    print(f"\n  Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
