"""
Training logger — captures detailed filter state at each evaluation step.
Saves to ../logs/{dataset}/{run_id}/ with JSON snapshots.
"""
import os
import json
import time
import torch
import numpy as np


class TrainLogger:
    def __init__(self, config, model):
        self.config = config
        self.model = None  # don't hold model reference (causes OOM on large datasets)
        self.enabled = config.get('log', False)
        if not self.enabled:
            return

        # Create log directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset = config['dataset']
        init = config['f_init']
        poly = config['poly']
        act = config['f_act']
        lr = config['lr']
        decay = config['decay']

        run_name = f"{dataset}_{poly}_K{config['f_order']}_{init}_{act}_lr{lr}_d{decay}_{timestamp}"
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
        self.log_dir = os.path.join(base_dir, dataset, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save config
        config_save = {k: str(v) for k, v in config.items()}
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config_save, f, indent=2)

        # Log eigenvalue info
        self._log_eigen_info(model)
        self.snapshots = []
        print(f"    Logging to: {self.log_dir}")

    def _log_eigen_info(self, model):
        """Log eigenvalue distributions at init."""
        info = {}
        if hasattr(model, 'user_eigenvals'):
            ev = model.user_eigenvals.cpu().numpy()
            info['user_eigenvals'] = {
                'values': ev.tolist(),
                'min': float(ev.min()), 'max': float(ev.max()),
                'mean': float(ev.mean()), 'std': float(ev.std()),
                'n': len(ev),
            }
        if hasattr(model, 'item_eigenvals'):
            ev = model.item_eigenvals.cpu().numpy()
            info['item_eigenvals'] = {
                'values': ev.tolist(),
                'min': float(ev.min()), 'max': float(ev.max()),
                'mean': float(ev.mean()), 'std': float(ev.std()),
                'n': len(ev),
            }
        with open(os.path.join(self.log_dir, 'eigen_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

    def log_epoch(self, epoch, loss, val_ndcg, val_recall, model):
        """Log filter state at evaluation step."""
        if not self.enabled:
            return

        snapshot = {
            'epoch': epoch,
            'loss': float(loss),
            'val_ndcg': float(val_ndcg),
            'val_recall': float(val_recall),
        }

        # Log filter coefficients and responses for each view
        for view_name, filt, eigenvals in [
            ('user', model.user_filter, getattr(model, 'user_eigenvals', None)),
            ('item', model.item_filter, getattr(model, 'item_eigenvals', None)),
        ]:
            if filt is None or eigenvals is None:
                continue

            # Raw coefficients
            coeffs = {}
            for name, param in filt.named_parameters():
                coeffs[name] = param.data.cpu().numpy().tolist()

            # Filter response on eigenvalues
            with torch.no_grad():
                response = filt(eigenvals).cpu().numpy()

            # Gradient magnitudes (if available)
            grads = {}
            for name, param in filt.named_parameters():
                if param.grad is not None:
                    g = param.grad.cpu().numpy()
                    grads[name] = {
                        'values': g.tolist(),
                        'norm': float(np.linalg.norm(g)),
                        'mean': float(g.mean()),
                        'max_abs': float(np.abs(g).max()),
                    }

            snapshot[f'{view_name}_filter'] = {
                'coefficients': coeffs,
                'response': response.tolist(),
                'response_min': float(response.min()),
                'response_max': float(response.max()),
                'response_mean': float(response.mean()),
                'response_std': float(response.std()),
                'gradients': grads,
            }

        # Fusion weights
        if hasattr(model, 'fusion_logits'):
            logits = model.fusion_logits.data.cpu().numpy()
            weights = torch.softmax(model.fusion_logits, dim=0).data.cpu().numpy()
            snapshot['fusion'] = {
                'logits': logits.tolist(),
                'weights': weights.tolist(),
            }

        self.snapshots.append(snapshot)

    def log_baseline(self, ndcg, recall, split='test'):
        """Log baseline result."""
        if not self.enabled:
            return
        baseline = {'ndcg': float(ndcg), 'recall': float(recall), 'split': split}
        filepath = os.path.join(self.log_dir, f'baseline_{split}.json')
        with open(filepath, 'w') as f:
            json.dump(baseline, f, indent=2)

    def log_final(self, best_epoch, baseline_ndcg, final_ndcg, baseline_recall, final_recall):
        """Log final results and save all snapshots."""
        if not self.enabled:
            return

        summary = {
            'best_epoch': best_epoch,
            'baseline_ndcg': float(baseline_ndcg),
            'final_ndcg': float(final_ndcg),
            'ndcg_improvement_pct': float((final_ndcg / baseline_ndcg - 1) * 100) if baseline_ndcg > 0 else 0,
            'baseline_recall': float(baseline_recall),
            'final_recall': float(final_recall),
            'n_snapshots': len(self.snapshots),
        }
        with open(os.path.join(self.log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # Save all epoch snapshots
        with open(os.path.join(self.log_dir, 'epochs.json'), 'w') as f:
            json.dump(self.snapshots, f, indent=2)

        # Save filter trajectory CSV for easy analysis
        self._save_trajectory_csv()

        print(f"    Log saved: {self.log_dir} ({len(self.snapshots)} snapshots)")

    def _save_trajectory_csv(self):
        """Save filter response trajectory as CSV for plotting."""
        if not self.snapshots:
            return

        rows = []
        for snap in self.snapshots:
            row = {
                'epoch': snap['epoch'],
                'loss': snap['loss'],
                'val_ndcg': snap['val_ndcg'],
            }
            for view in ['user', 'item']:
                key = f'{view}_filter'
                if key in snap:
                    resp = snap[key]['response']
                    for i, v in enumerate(resp):
                        row[f'{view}_h_{i}'] = v
                    row[f'{view}_response_mean'] = snap[key]['response_mean']
                    row[f'{view}_response_std'] = snap[key]['response_std']

                    # Gradient norm
                    grads = snap[key].get('gradients', {})
                    total_grad = 0
                    for g in grads.values():
                        total_grad += g.get('norm', 0)
                    row[f'{view}_grad_norm'] = total_grad

            rows.append(row)

        import csv
        filepath = os.path.join(self.log_dir, 'trajectory.csv')
        if rows:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
