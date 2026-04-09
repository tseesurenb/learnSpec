"""
Filter search: find best spectral filter configuration per dataset.

Stage 1 (--stage 1): Infer-only search over f_poly × f_init × f_order × f_act
Stage 2 (--stage 2): Training search over lr × decay on top configs from Stage 1

Usage:
    python search_filter.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --beta 0.5 --stage 1
    python search_filter.py --dataset ml-100k --u_eigen 25 --i_eigen 130 --beta 0.5 --stage 2
"""

import argparse
import os
import csv
import time
from itertools import product
from main import main as run_experiment

INITS = ['uniform', 'lowpass', 'highpass', 'bandpass', 'butterworth', 'bandreject', 'decay', 'rise', 'plateau']
DIRECT_INITS = ['uniform', 'lowpass', 'highpass', 'bandpass']
ORDERS = [4, 8, 16, 24, 32]
ACTIVATIONS = ['sigmoid', 'softplus', 'tanh', 'none']
POLYS = ['bernstein', 'cheby', 'direct']

LRS = [0.0005, 0.001, 0.002, 0.005]
DECAYS = [0.001, 0.01, 0.1]


def parse_args():
    parser = argparse.ArgumentParser(description='Filter search')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--u_eigen', type=int, required=True)
    parser.add_argument('--i_eigen', type=int, required=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='all', choices=['all', 'bernstein', 'cheby', 'direct'],
                        help='Filter type to search: all, bernstein, cheby, or direct')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2], help='1=infer search, 2=training search')
    parser.add_argument('--top_k', type=int, default=5, help='Top configs from stage 1 to use in stage 2')
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def build_stage1_configs(args):
    """Generate all (poly, init, order, act) combinations for infer search."""
    polys = POLYS if args.model == 'all' else [args.model]
    configs = []
    for poly in polys:
        if poly == 'direct':
            for init, act in product(DIRECT_INITS, ACTIVATIONS):
                configs.append({
                    'dataset': args.dataset, 'seed': 42, 'view': 'ui',
                    'u_eigen': args.u_eigen, 'i_eigen': args.i_eigen, 'beta': args.beta,
                    'f_order': 32, 'f_init': init, 'poly': poly,
                    'f_dropout': 0.0, 'f_act': act,
                    'opt': 'rmsprop', 'lr': 0.001, 'decay': 0.01,
                    'epochs': 200, 'batch_size': 1024,
                    'patience': 10, 'eval_every': 5,
                    'split_ratio': 0.7,
                    'infer': True, 'save': False,
                    'device': args.device, 'topks': [20],
                })
        else:
            for init, order, act in product(INITS, ORDERS, ACTIVATIONS):
                configs.append({
                    'dataset': args.dataset, 'seed': 42, 'view': 'ui',
                    'u_eigen': args.u_eigen, 'i_eigen': args.i_eigen, 'beta': args.beta,
                    'f_order': order, 'f_init': init, 'poly': poly,
                    'f_dropout': 0.0, 'f_act': act,
                    'opt': 'rmsprop', 'lr': 0.001, 'decay': 0.01,
                    'epochs': 200, 'batch_size': 1024,
                    'patience': 10, 'eval_every': 5,
                    'split_ratio': 0.7,
                    'infer': True, 'save': False,
                    'device': args.device, 'topks': [20],
                })
    return configs


def build_stage2_configs(args, top_configs):
    """Generate training configs from top stage 1 results."""
    configs = []
    for base in top_configs:
        for lr, decay in product(LRS, DECAYS):
            cfg = base.copy()
            cfg['infer'] = False
            cfg['lr'] = lr
            cfg['decay'] = decay
            cfg['epochs'] = 300
            cfg['patience'] = 10
            configs.append(cfg)
    return configs


def load_stage1_results(csv_path, top_k=5):
    """Load top configs from stage 1 CSV. Includes best per poly type + overall top_k."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['ndcg'] = float(row['ndcg'])
            row['recall'] = float(row['recall'])
            results.append(row)
    results.sort(key=lambda x: x['ndcg'], reverse=True)

    # Overall top_k
    selected = results[:top_k]

    # Force-include best per poly type
    seen_polys = {r['f_poly'] for r in selected}
    for poly in POLYS:
        if poly not in seen_polys:
            for r in results:
                if r['f_poly'] == poly:
                    selected.append(r)
                    print(f"  Force-included best {poly}: {r['f_init']}, K={r['f_order']}, {r['f_act']} → NDCG={r['ndcg']:.4f}")
                    break

    top_configs = []
    seen = set()
    for r in selected:
        key = (r['f_poly'], r['f_init'], r['f_order'], r['f_act'])
        if key in seen:
            continue
        seen.add(key)
        top_configs.append({
            'dataset': r['dataset'], 'seed': 42, 'view': 'ui',
            'u_eigen': int(r['u_eigen']), 'i_eigen': int(r['i_eigen']), 'beta': float(r['beta']),
            'f_order': int(r['f_order']), 'f_init': r['f_init'], 'poly': r['f_poly'],
            'f_dropout': 0.0, 'f_act': r['f_act'],
            'opt': 'rmsprop', 'batch_size': 1024, 'eval_every': 5,
            'split_ratio': 0.7, 'save': False,
            'device': r.get('device', 'auto'), 'topks': [20],
        })

    print(f"  Selected {len(top_configs)} configs for stage 2 training")
    return top_configs


def run_stage1(args):
    import torch
    import utils as ut
    from model import LearnSpecCF
    from filter import create_filter
    import procedure as pr

    configs = build_stage1_configs(args)
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'search_stage1_{args.model}_{args.dataset}.csv')

    fieldnames = ['dataset', 'u_eigen', 'i_eigen', 'beta', 'f_poly', 'f_init', 'f_order', 'f_act', 'ndcg', 'recall']

    # Resume: load existing results
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['f_poly'], row['f_init'], row['f_order'], row['f_act'])
                done.add(key)
        print(f"Resuming: {len(done)} configs already done")
        f_out = open(csv_path, 'a', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        f_out = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

    total = len(configs)
    skipped = 0
    print(f"Stage 1: {total} configs for {args.dataset}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using: {device}")

    # Load dataset and base model ONCE
    base_config = configs[0].copy()
    base_config['device'] = device
    ut.set_seed(42)
    dataset = ut.load_dataset(base_config)

    print("Loading eigendecompositions (once)...")
    base_model = LearnSpecCF(dataset.UserItemNet, base_config, use_cache=True).to(device)

    # Cache eigenvecs/eigenvals and spectral projections
    cached = {
        'user_eigenvals': base_model.user_eigenvals if hasattr(base_model, 'user_eigenvals') else None,
        'user_eigenvecs': base_model.user_eigenvecs if hasattr(base_model, 'user_eigenvecs') else None,
        'item_eigenvals': base_model.item_eigenvals if hasattr(base_model, 'item_eigenvals') else None,
        'item_eigenvecs': base_model.item_eigenvecs if hasattr(base_model, 'item_eigenvecs') else None,
        'user_spectral_R': base_model.user_spectral_R if hasattr(base_model, 'user_spectral_R') else None,
        'item_spectral_R': base_model.item_spectral_R if hasattr(base_model, 'item_spectral_R') else None,
    }
    print("Eigendecompositions cached. Starting search...\n")

    run_count = 0
    for i, cfg in enumerate(configs):
        key = (cfg['poly'], cfg['f_init'], str(cfg['f_order']), cfg['f_act'])
        if key in done:
            skipped += 1
            continue

        cfg['device'] = device

        try:
            # Create new filters without reloading eigen data
            view_config_u = cfg.copy()
            view_config_u['n_eigen'] = args.u_eigen
            view_config_i = cfg.copy()
            view_config_i['n_eigen'] = args.i_eigen

            user_filter = create_filter(order=cfg['f_order'], init_type=cfg['f_init'], config=view_config_u).to(device)
            item_filter = create_filter(order=cfg['f_order'], init_type=cfg['f_init'], config=view_config_i).to(device)

            # Swap filters on the base model
            base_model.user_filter = user_filter
            base_model.item_filter = item_filter
            base_model.eval()

            with torch.no_grad():
                results = pr.evaluate(dataset, base_model, split='test', batch_size=cfg['batch_size'])
            ndcg = results['ndcg'][0]
            recall = results['recall'][0]
        except Exception as e:
            print(f"  ERROR: {e}")
            ndcg, recall = 0.0, 0.0

        row = {
            'dataset': args.dataset, 'u_eigen': args.u_eigen, 'i_eigen': args.i_eigen,
            'beta': args.beta, 'f_poly': cfg['poly'], 'f_init': cfg['f_init'],
            'f_order': cfg['f_order'], 'f_act': cfg['f_act'],
            'ndcg': f'{ndcg:.4f}', 'recall': f'{recall:.4f}',
        }
        writer.writerow(row)
        f_out.flush()

        run_count += 1
        print(f"[{run_count}/{total - len(done)}] {cfg['poly']}(K={cfg['f_order']}) | {cfg['f_init']} | {cfg['f_act']} → NDCG={ndcg:.4f}, Recall={recall:.4f}")

    f_out.close()
    del base_model

    # Print top results
    print(f"\n{'='*60}")
    print(f"Stage 1 complete: {csv_path}")
    print_top_results(csv_path)


def run_stage2(args):
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    stage1_path = os.path.join(results_dir, f'search_stage1_{args.model}_{args.dataset}.csv')

    if not os.path.exists(stage1_path):
        print(f"ERROR: Stage 1 results not found: {stage1_path}")
        print(f"Run stage 1 first: python search_filter.py --dataset {args.dataset} --stage 1 ...")
        return

    top_configs = load_stage1_results(stage1_path, top_k=args.top_k)
    configs = build_stage2_configs(args, top_configs)

    csv_path = os.path.join(results_dir, f'search_stage2_{args.model}_{args.dataset}.csv')
    fieldnames = ['dataset', 'f_poly', 'f_init', 'f_order', 'f_act', 'lr', 'decay',
                  'best_epoch', 'baseline_ndcg', 'ndcg', 'ndcg_pct', 'recall']

    # Resume
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['f_poly'], row['f_init'], row['f_order'], row['f_act'], row['lr'], row['decay'])
                done.add(key)
        print(f"Resuming: {len(done)} configs already done")
        f_out = open(csv_path, 'a', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        f_out = open(csv_path, 'w', newline='')
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

    import torch
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    total = len(configs)
    skipped = 0
    print(f"Stage 2: {total} training configs for {args.dataset}")

    for i, cfg in enumerate(configs):
        key = (cfg['poly'], cfg['f_init'], str(cfg['f_order']), cfg['f_act'], str(cfg['lr']), str(cfg['decay']))
        if key in done:
            skipped += 1
            continue

        cfg['device'] = device

        try:
            result = run_experiment(config_override=cfg)
            ndcg = result['ndcg']
            recall = result['recall']
            baseline = result['baseline_ndcg']
            best_epoch = result['best_epoch']
            ndcg_pct = result['ndcg_improvement_pct']
        except Exception as e:
            print(f"  ERROR: {e}")
            ndcg, recall, baseline, best_epoch, ndcg_pct = 0.0, 0.0, 0.0, 0, 0.0

        row = {
            'dataset': args.dataset, 'f_poly': cfg['poly'], 'f_init': cfg['f_init'],
            'f_order': cfg['f_order'], 'f_act': cfg['f_act'],
            'lr': cfg['lr'], 'decay': cfg['decay'],
            'best_epoch': best_epoch, 'baseline_ndcg': f'{baseline:.4f}',
            'ndcg': f'{ndcg:.4f}', 'ndcg_pct': f'{ndcg_pct:+.1f}',
            'recall': f'{recall:.4f}',
        }
        writer.writerow(row)
        f_out.flush()

        rank = i - skipped + 1
        print(f"[{rank}/{total - skipped}] {cfg['poly']}(K={cfg['f_order']}) | {cfg['f_init']} | {cfg['f_act']} | "
              f"lr={cfg['lr']}, decay={cfg['decay']} → NDCG={ndcg:.4f} ({ndcg_pct:+.1f}%)")

    f_out.close()

    print(f"\n{'='*60}")
    print(f"Stage 2 complete: {csv_path}")
    print_top_results(csv_path, key='ndcg')


def print_top_results(csv_path, key='ndcg', top_k=10):
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['ndcg'] = float(row['ndcg'])
            row['recall'] = float(row['recall'])
            results.append(row)

    results.sort(key=lambda x: x[key], reverse=True)
    print(f"\nTop {top_k} results:")
    print(f"{'Rank':>4} | {'Poly':>9} | {'Init':>10} | {'K':>3} | {'Act':>8} | {'NDCG':>7} | {'Recall':>7}")
    print("-" * 65)
    for i, r in enumerate(results[:top_k]):
        print(f"{i+1:>4} | {r.get('f_poly',''):>9} | {r.get('f_init',''):>10} | {r.get('f_order',''):>3} | "
              f"{r.get('f_act',''):>8} | {r['ndcg']:.4f} | {r['recall']:.4f}")


if __name__ == '__main__':
    args = parse_args()
    if args.stage == 1:
        run_stage1(args)
    elif args.stage == 2:
        run_stage2(args)
