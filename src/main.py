# LearnSpec: Spectral Collaborative Filtering for Recommender Systems
# Author: Tseesuren Batsuuri, Macquarie University, 2026/04/08

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import gc

from model import LearnSpecCF
from config import parse_args, get_config, SPLIT_SEED
import procedure as pr
import utils as ut
from utils import C
from train_logger import TrainLogger


def main(config_override=None):
    if config_override and 'dataset' in config_override:
        # Direct config — skip argparse (used by search scripts)
        config = config_override.copy()
        if not isinstance(config.get('device'), torch.device):
            if config.get('device') == 'auto' or config.get('device') is None:
                config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                config['device'] = torch.device(config['device'])
    else:
        args = parse_args()
        config = get_config(args)
        if config_override:
            config.update(config_override)
    ut.set_seed(config['seed'])

    quiet = config.get('quiet', 1)
    sr = config.get('split_ratio', 0.7)
    poly_info = f"{config['poly']}(u={config['u_eigen']},i={config['i_eigen']})" if config['poly'] == 'direct' else f"{config['poly']}(K={config['f_order']})"
    # Fixed-width tag for aligned output
    ds = f"{config['dataset']:<12s}"
    pi = f"{poly_info:<20s}"
    init = f"{config['f_init']:<10s}"
    hp = f"β={config['beta']:<4} lr={config['lr']:<6} d={config['decay']:<5}"
    tag = f"{ds} {pi} {init} {hp}"

    if quiet == 0:
        print(f"LearnSpec: {config['dataset']}({sr}) | {config['view']} | {poly_info} | "
              f"init={config['f_init']} | act={config['f_act']} | "
              f"u={config['u_eigen']},i={config['i_eigen']} | beta={config['beta']} | "
              f"{config.get('loss','bpr').upper()} | {config['opt']}(lr={config['lr']}, decay={config['decay']}) | "
              f"patience={config['patience']} | {config['device']}")

    dataset = ut.load_dataset(config)

    if quiet == 0:
        print("Computing baseline...")
    baseline_ndcg, baseline_recall = pr.evaluate_baseline(dataset, config)
    if quiet == 0:
        print(f"{C.B}{C.BOLD}Baseline (test): NDCG={baseline_ndcg:.4f}, Recall={baseline_recall:.4f}{C.END}")

    if config.get('infer', False):
        return {'baseline_ndcg': baseline_ndcg, 'baseline_recall': baseline_recall}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    split_ratio = config.get('split_ratio', 0.7)
    partial_train, validation_data = ut.split_training_data(dataset, split_ratio=split_ratio, seed=SPLIT_SEED)
    partial_adj = ut.create_partial_adj_matrix(partial_train, dataset.n_users, dataset.m_items)

    model = LearnSpecCF(partial_adj, config, use_cache=True, split_seed=SPLIT_SEED, split_ratio=split_ratio, verbose=(quiet==0)).to(config['device'])
    optimizer = ut.create_optimizer(config, model.get_optimizer_groups())
    initial_params = {'config': config, **model.get_filter_snapshot()} if config['save'] else None
    logger = TrainLogger(config, model)

    model.eval()
    with torch.no_grad():
        temp_ds = ut.create_temp_dataset(validation_data, dataset, partial_train)
        val_results = pr.evaluate(temp_ds, model, split='val', batch_size=config['batch_size'])
    val_baseline_ndcg = val_results['ndcg'][0]
    if quiet == 0:
        print(f"{C.G}{C.BOLD}Baseline (validation): NDCG@20={val_baseline_ndcg:.4f}, Recall@20={val_results['recall'][0]:.4f}{C.END}")
    logger.log_baseline(baseline_ndcg, baseline_recall, 'test')
    logger.log_baseline(val_baseline_ndcg, val_results['recall'][0], 'val')
    logger.log_epoch(0, 0.0, val_baseline_ndcg, val_results['recall'][0], model)

    best_ndcg, best_recall, best_epoch = 0, 0, -1
    best_state = None
    patience_counter = 0
    prev_params = ut.get_current_parameters(model)
    epoch_snapshots = []
    eval_every = config.get('eval_every', 5)
    ew = len(str(config['epochs']))

    for epoch in range(config['epochs']):
        model.train()
        loss = pr.train_spectral(validation_data, model, optimizer,
                                 batch_size=config['batch_size'],
                                 f_reg=config.get('f_reg', 0.0),
                                 loss_type=config.get('loss', 'bpr'))

        if (epoch + 1) % eval_every != 0 and epoch > 0:
            if quiet == 0:
                print(f"\rEpoch {epoch+1:0{ew}d}/{config['epochs']} | Loss: {loss:.4f}    ", end='', flush=True)
            else:
                pct = (epoch + 1) / config['epochs']
                filled = int(pct * 15)
                bar = '■' * filled + '·' * (15 - filled)
                print(f"\r{C.DIM}{tag}{C.END}  {bar} {epoch+1:>4d}/{config['epochs']}          ", end='', flush=True)
            continue

        model.eval()
        with torch.no_grad():
            temp_ds = ut.create_temp_dataset(validation_data, dataset, partial_train)
            results = pr.evaluate(temp_ds, model, split='val', batch_size=config['batch_size'])
        ndcg, recall = results['ndcg'][0], results['recall'][0]

        param_changes = ut.get_parameter_changes(model, prev_params)
        is_best = ndcg > best_ndcg
        if quiet == 0:
            parts = [f"Epoch {epoch+1:0{ew}d}/{config['epochs']}", f"Loss: {loss:.4f}",
                     f"NDCG@20: {ndcg:.4f}", f"Recall@20: {recall:.4f}",
                     f"Δ_NDCG: {ndcg - val_baseline_ndcg:+.4f}"]
            if param_changes:
                parts.append(f"Δ_params: {np.mean([c['mean_change'] for c in param_changes.values()]):.6f}")
            print("\r" + " | ".join(parts) + (" *" if is_best else ""))
        else:
            pct = (epoch + 1) / config['epochs']
            filled = int(pct * 15)
            bar = '■' * filled + '·' * (15 - filled)
            star = ' ★' if is_best else ''
            print(f"\r{C.DIM}{tag}{C.END}  {bar} {epoch+1:>4d}/{config['epochs']}  NDCG:{ndcg:.4f} R:{recall:.4f}{star}     ", end='', flush=True)
        prev_params = ut.get_current_parameters(model)

        logger.log_epoch(epoch + 1, loss, ndcg, recall, model)

        if is_best:
            best_ndcg, best_recall, best_epoch = ndcg, recall, epoch + 1
            best_state = model.get_filter_snapshot()
            patience_counter = 0
        else:
            patience_counter += 1

        if config['save']:
            snapshot = model.get_filter_snapshot()
            snapshot.update({'epoch': epoch + 1, 'val_ndcg': ndcg, 'val_recall': recall})
            epoch_snapshots.append(snapshot)

        if patience_counter >= config['patience']:
            break

    pass  # quiet newline handled in final output

    if best_state is None:
        best_state = model.get_filter_snapshot()

    # Free all training memory before loading final model
    del model, partial_train, validation_data, partial_adj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_model = LearnSpecCF(dataset.UserItemNet, config, use_cache=True, verbose=(quiet==0)).to(config['device'])
    with torch.no_grad():
        final_model.load_filter_snapshot(best_state)
        final_model.eval()
        r = pr.evaluate(dataset, final_model, split='test', batch_size=config['batch_size'])
    final_ndcg, final_recall = r['ndcg'][0], r['recall'][0]
    logger.log_final(best_epoch, baseline_ndcg, final_ndcg, baseline_recall, final_recall)

    if config['save']:
        params_file = ut.save_run_results(
            config, initial_params, epoch_snapshots, best_state,
            best_epoch, best_ndcg, final_ndcg, final_recall, baseline_ndcg, baseline_recall)
        print(f"Filter parameters saved to: {params_file}")

    ndcg_pct = (final_ndcg / baseline_ndcg - 1) * 100
    recall_pct = (final_recall / baseline_recall - 1) * 100
    if quiet == 1:
        nc = C.G + C.BOLD if ndcg_pct > 0 else C.R + C.BOLD
        rc = C.G + C.BOLD if recall_pct > 0 else C.R + C.BOLD
        # Clear the line fully then print final result
        print(f"\r{' ' * 160}\r", end='')
        print(f"{C.DIM}{tag}{C.END}  ep {best_epoch:>4d}  NDCG@20 {C.BOLD}{baseline_ndcg:.4f}{C.END}→{nc}{final_ndcg:.4f}{C.END}({nc}{ndcg_pct:+.1f}%{C.END})  Recall@20 {C.BOLD}{baseline_recall:.4f}{C.END}→{rc}{final_recall:.4f}{C.END}({rc}{recall_pct:+.1f}%{C.END})")
    else:
        print(f"\n{C.BOLD}RESULTS (best epoch {best_epoch}):{C.END}")
        poly_info2 = f"{config['poly']}(u={config['u_eigen']},i={config['i_eigen']})" if config['poly'] == 'direct' else f"{config['poly']}(K={config['f_order']})"
        print(f"Config:   {config['dataset']}({config.get('split_ratio',0.7)}) | {poly_info2} | init={config['f_init']} | act={config['f_act']} | "
              f"u={config['u_eigen']},i={config['i_eigen']} | beta={config['beta']} | "
              f"{config.get('loss','bpr').upper()} | {config['opt']}(lr={config['lr']}, decay={config['decay']})")
        print(f"{C.B}{C.BOLD}Baseline: NDCG={baseline_ndcg:.4f}, Recall={baseline_recall:.4f}{C.END}")
        print(f"Final:    {C.G if ndcg_pct > 0 else C.R}NDCG={final_ndcg:.4f} ({ndcg_pct:+.1f}%){C.END}, "
              f"{C.G if recall_pct > 0 else C.R}Recall={final_recall:.4f} ({recall_pct:+.1f}%){C.END}")

    return {
        'best_epoch': best_epoch,
        'val_ndcg': best_ndcg, 'val_recall': best_recall,
        'baseline_ndcg': baseline_ndcg, 'baseline_recall': baseline_recall,
        'ndcg': final_ndcg, 'recall': final_recall,
        'ndcg_improvement_pct': ndcg_pct, 'recall_improvement_pct': recall_pct,
    }


if __name__ == "__main__":
    main()
