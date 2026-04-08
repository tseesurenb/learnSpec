import sys
import os
import torch
import gc
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import Dataset
from model import LearnSpecCF
from procedure import Test


def get_available_betas(cache_dir, dataset_name):
    betas = set()
    if not os.path.exists(cache_dir):
        return [0.5]
    for f in os.listdir(cache_dir):
        if f.startswith(f"full_{dataset_name}_") and 'degNorm_' in f and (f.endswith('.pkl') or f.endswith('.npz')):
            try:
                beta_str = f.split('degNorm_')[1].split('_')[0].split('.')[0]
                betas.add(float(beta_str.replace('p', '.')))
            except (ValueError, IndexError):
                continue
    return sorted(betas) if betas else [0.5]


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


def make_config(dataset_name, u_eigen, i_eigen, beta, device):
    return {
        'dataset': dataset_name, 'seed': 42, 'view': 'ui',
        'u_eigen': u_eigen, 'i_eigen': i_eigen, 'beta': beta,
        'f_order': 8, 'f_init': 'uniform', 'poly': 'bernstein',
        'f_dropout': 0.0, 'f_act': 'sigmoid',
        'opt': 'rmsprop', 'lr': 0.01, 'decay': 1e-5,
        'epochs': 0, 'batch_size': 1000, 'patience': 5,
        'device': device, 'loss': 'mse', 'topks': [20],
    }


def load_dataset_once(dataset_name, config):
    return Dataset(path=f"../data/{dataset_name}")


def eval_config(dataset, config, device):
    try:
        model = LearnSpecCF(dataset.UserItemNet, config, use_cache=True, verbose=False).to(device)
        model.eval()
        with torch.no_grad():
            results = Test(dataset, model, epoch=-1)
        ndcg, recall = results['ndcg'][0], results['recall'][0]
        del model; gc.collect()
        return ndcg, recall
    except Exception:
        return None


def search_beta(dataset_name, betas, device, u_eigen=100, i_eigen=100):
    print(f"\n  {'='*60}")
    print(f"  Stage 1: Beta search (u_eigen={u_eigen}, i_eigen={i_eigen})")
    print(f"  {'='*60}")
    print(f"  {'beta':>8} {'NDCG':>8} {'Recall':>8}")
    print(f"  {'-'*28}")

    best_ndcg, best_beta = 0, betas[0]
    results = []

    for beta in betas:
        config = make_config(dataset_name, u_eigen, i_eigen, beta, device)
        dataset = load_dataset_once(dataset_name, config)
        result = eval_config(dataset, config, device)
        del dataset

        if result is None:
            print(f"  {beta:>8.2f}   FAILED")
            continue

        ndcg, recall = result
        improved = ndcg > best_ndcg
        print(f"  {beta:>8.2f} {ndcg:>8.4f} {recall:>8.4f}{' *' if improved else ''}")
        results.append({'beta': beta, 'ndcg': ndcg, 'recall': recall})

        if improved:
            best_ndcg, best_beta = ndcg, beta

    print(f"  >> Best beta={best_beta} (NDCG={best_ndcg:.4f})")
    return best_beta, results


def search_eigen_grid(dataset_name, beta, device, step=50, max_u=None, max_i=None):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', dataset_name)
    avail_max_u = get_max_eigen(cache_dir, dataset_name, 'u', beta)
    avail_max_i = get_max_eigen(cache_dir, dataset_name, 'i', beta)
    if max_u: avail_max_u = min(avail_max_u, max_u)
    if max_i: avail_max_i = min(avail_max_i, max_i)
    if avail_max_u == 0 or avail_max_i == 0:
        print(f"\n  Beta={beta}: no eigendecompositions found, skipping")
        return [], [], []

    u_values = list(range(step, avail_max_u + 1, step))
    i_values = list(range(step, avail_max_i + 1, step))
    if not u_values or not i_values:
        return [], [], []

    print(f"\n  {'='*60}")
    print(f"  Beta = {beta}, u: {u_values[0]}..{u_values[-1]}, i: {i_values[0]}..{i_values[-1]}")
    print(f"  {len(u_values) * len(i_values)} evaluations")
    print(f"  {'='*60}")
    print(f"  {'u_eigen':>8} {'i_eigen':>8} {'NDCG':>8} {'Recall':>8}")
    print(f"  {'-'*36}")

    config = make_config(dataset_name, step, step, beta, device)
    dataset = load_dataset_once(dataset_name, config)
    all_results = []
    best_grid_ndcg = 0
    u_no_improve, prev_u_best = 0, 0

    for u_e in u_values:
        row_best, i_no_improve = 0, 0
        for i_e in i_values:
            cfg = make_config(dataset_name, u_e, i_e, beta, device)
            result = eval_config(dataset, cfg, device)
            if result:
                ndcg, recall = result
                improved = ndcg > best_grid_ndcg
                if improved: best_grid_ndcg = ndcg
                all_results.append({'u_eigen': u_e, 'i_eigen': i_e, 'beta': beta, 'ndcg': ndcg, 'recall': recall})
                print(f"  {u_e:>8} {i_e:>8} {ndcg:>8.4f} {recall:>8.4f}{' *' if improved else ''}", flush=True)
                if ndcg > row_best:
                    row_best, i_no_improve = ndcg, 0
                else:
                    i_no_improve += 1
                    if i_no_improve >= 3: break
            else:
                break

        if row_best > prev_u_best:
            prev_u_best, u_no_improve = row_best, 0
        else:
            u_no_improve += 1
            if u_no_improve >= 3: break

    del dataset; gc.collect()
    return all_results, u_values, i_values


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--eigen_step', type=int, default=50)
    parser.add_argument('--max_u', type=int, default=None)
    parser.add_argument('--max_i', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {'GPU: ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'runs', 'search', f'{args.dataset}_search_{timestamp}.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    betas = get_available_betas(cache_dir, args.dataset)
    print(f"\n{'='*70}")
    print(f"  Baseline Search: {args.dataset} | betas: {betas}")
    print(f"{'='*70}")

    save_data = {'dataset': args.dataset, 'timestamp': timestamp}
    all_beta_results = []

    for beta in betas:
        grid_results, u_vals, i_vals = search_eigen_grid(
            args.dataset, beta, device, step=args.eigen_step, max_u=args.max_u, max_i=args.max_i)
        if not grid_results:
            continue
        best = max(grid_results, key=lambda x: x['ndcg'])
        best['beta'] = beta
        all_beta_results.append(best)
        save_data[f'grid_beta_{beta}'] = grid_results
        print(f"\n  ** Beta={beta} best: u={best['u_eigen']}, i={best['i_eigen']} => NDCG={best['ndcg']:.4f}")

    all_beta_results.sort(key=lambda x: x['ndcg'], reverse=True)
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: {args.dataset}")
    print(f"{'='*70}")
    print(f"  {'beta':>6} {'u_eigen':>8} {'i_eigen':>8} {'NDCG':>8} {'Recall':>8}")
    print(f"  {'-'*42}")
    for i, r in enumerate(all_beta_results):
        print(f"  {r['beta']:>6.2f} {r['u_eigen']:>8} {r['i_eigen']:>8} {r['ndcg']:>8.4f} {r['recall']:>8.4f}{'  <-- BEST' if i == 0 else ''}")

    if all_beta_results:
        b = all_beta_results[0]
        print(f"\n  Best command:")
        print(f"    python main.py --dataset {args.dataset} --view ui --beta {b['beta']} "
              f"--u_eigen {b['u_eigen']} --i_eigen {b['i_eigen']}")

    save_data['summary'] = all_beta_results
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")


if __name__ == '__main__':
    main()
