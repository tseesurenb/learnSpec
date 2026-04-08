'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import torch
import numpy as np
import torch.nn.functional as F
import utils as ut
from tqdm import tqdm
from model import RecSysGNN
from config import config, C
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def compute_bpr_loss(users, u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0, model=None):
    # Standard BPR loss for SimGCF and LightGCN
    pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
    
    if config['samples'] == 1:
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores + config['margin']))
        reg_loss = (u_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2)) / (2 * len(users))
    else:
        neg_scores = torch.sum(u_emb.unsqueeze(1) * neg_emb, dim=2)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores.unsqueeze(1) + config['margin']))
        reg_loss = (u_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2, dim=2).pow(2).sum() / neg_emb0.shape[1]) / (2 * len(users))
    
    return bpr_loss, reg_loss, None

def create_lr_scheduler(optimizer, stype):
    if stype == 'step':
        return StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_factor'])
    elif stype == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=config['lr_factor'], patience=config['lr_patience'], min_lr=config['min_lr'])
    return None


def BPR_train(data, model, optimizer, epoch, g_seed=42, pbar=None):
    model.train()
    losses = []
    
    
    i_neg = torch.tensor(data.sample_negatives(epoch, g_seed), device=data.device, dtype=torch.long)
    n_batches = len(data.train_u_t) // config['batch'] + 1
    graph_info = "knn" if config['edge'] == 'knn' else "bi"
    
    for b_i, (b_u, b_i_pos, b_i_neg) in enumerate(ut.minibatch(data.train_u_t, data.train_i_pos_t, i_neg, batch_size=config['batch'])):
        optimizer.zero_grad()
        u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0 = model.encode_minibatch(b_u, b_i_pos, b_i_neg, data)
        bpr_loss, reg_loss, _ = compute_bpr_loss(b_u, u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0, model)
        total_loss = bpr_loss + config['decay'] * reg_loss
        total_loss.backward()
        optimizer.step()
        loss_val = total_loss.detach().item()
        losses.append(loss_val)
        
        if pbar:
            pbar.set_description(f"{config['model']}-{graph_info}({g_seed:4}) | ep({config['epochs']:3}) {epoch:3} | ba({n_batches:3}) {b_i:3} | loss {loss_val:.4f} | {ut.mem_usage(data.device)}")
    
    return np.mean(losses)

def Test(data, model, epoch=None, K=20):
    model.eval()
    with torch.no_grad():
        # For embedding-based models (SimGCF, LightGCN)
        _, out = model(data)
        u_emb, i_emb = torch.split(out, (data.u_n, data.i_n))
        # Get metrics for multiple K values
        metrics = ut.get_metrics_multi_k(u_emb, i_emb, data, K_list=[5, 10, 20], batch_size=config['batch'])
        # Also get single K for backward compatibility
        recall, precision, ndcg = ut.get_metrics(u_emb, i_emb, data, K, batch_size=config['batch'])
        metrics.update({'recall': recall, 'precision': precision, 'ndcg': ndcg})
    return metrics

def run_experiment(dataset_name, exp_n=1, g_seed=42, device='cpu', verbose=-1):
    import time
    from data import Data
    
    start_time = time.time()
    run = ut.init_wandb()
    data = Data(dataset_name, device)
    
    if verbose >= 0:
        sim_info = f"K: {config['u_K']}-{config['i_K']} | " if config['edge'] == 'knn' else ""
        graph_type = config['edge']
        print("-" * 220)
        print(f"data: {C.BLUE}{dataset_name}{C.RESET} (u-{C.RED}{data.u_n}{C.RESET}, i-{C.RED}{data.i_n}{C.RESET}, train-{data.train_n}, test-{data.test_n}) | model: {C.GREEN}{config['model']}{C.RESET} | seed: {g_seed} | L: {config['layers']} | {sim_info}batch: {config['batch']} | lr: {config['lr']} | decay: {config['decay']} | margin: {config['margin']} | samples: {config['samples']} | emb: {config['emb_dim']} | n_temp: {config['norm_temp']} | graph: {C.GREEN}{graph_type}{C.RESET}")
        if verbose != 2:
            print("-" * 220)
    
    model = RecSysGNN(model=config['model'], emb_dim=config['emb_dim'], l_n=config['layers'], 
                      u_n=data.u_n, i_n=data.i_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = create_lr_scheduler(opt, config['lr_sched'])
    
    losses = {'bpr_loss': []}
    metrics = {'recall': [], 'precision': [], 'f1': [], 'ncdg': []}
    best_ncdg = best_recall = best_epoch = 0
    
    pbar = tqdm(range(config['epochs']), bar_format='{desc}{bar:20} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    best_prec = 0
    
    for epoch in pbar:
        avg_loss = BPR_train(data, model, opt, epoch, g_seed, pbar)
        losses['bpr_loss'].append(round(avg_loss, 4))
        
        if epoch % config["eval_freq"] == 0 and epoch > 0:
            results = Test(data, model, epoch, K=config['eval_K'])
            recall, prec, ncdg = results['recall'], results['precision'], results['ndcg']
            
            # Extract multi-K metrics
            ndcg5, ndcg10, hr5, hr10 = results['ndcg@5'], results['ndcg@10'], results['hr@5'], results['hr@10']
            
            if ncdg > best_ncdg or (ncdg == best_ncdg and recall >= best_recall):
                best_ncdg, best_recall, best_prec, best_epoch = ncdg, recall, prec, epoch
            
            if scheduler:
                old_lr = opt.param_groups[0]['lr']
                if config['lr_sched'] == 'plateau':
                    scheduler.step(ncdg)
                else:
                    scheduler.step()
                
                # Enforce minimum learning rate
                for param_group in opt.param_groups:
                    if param_group['lr'] < 0.001:
                        param_group['lr'] = 0.001
                
                new_lr = opt.param_groups[0]['lr']
                if verbose >= 1 and old_lr != new_lr:
                    print(f"\n{C.BLUE}LR changed: {old_lr:.2e} → {new_lr:.2e}{C.RESET}")
            
            current_lr = opt.param_groups[0]['lr']
            pbar.set_postfix_str(f"N@5:{ndcg5:.3f} N@10:{ndcg10:.3f} H@5:{hr5:.3f} H@10:{hr10:.3f} (best N@20:{best_ncdg:.4f} @ep{best_epoch}, lr {current_lr:.2e})")
            
            if verbose == 1:
                pbar.display()
                print()
            elif verbose == 2:
                # Compact progress update for verbose=2
                print(f"\rEpoch {epoch:3}/{config['epochs']:3} | Recall: {recall:.4f} | NDCG: {ncdg:.4f} (best: {best_ncdg:.4f}@{best_epoch}) | LR: {current_lr:.2e}", end="", flush=True)
            
            f1 = (2 * recall * prec / (recall + prec)) if (recall + prec) else 0.0
            metrics['recall'].append(round(recall, 4))
            metrics['precision'].append(round(prec, 4))
            metrics['f1'].append(round(f1, 4))
            metrics['ncdg'].append(round(ncdg, 4))
            
            if config['wandb']:
                wandb_metrics = {"ncdg": ncdg, "recall@20": recall, "bpr_loss": avg_loss}
                wandb_metrics.update({k: v for k, v in results.items() if k.startswith(('ndcg@', 'hr@', 'recall@', 'precision@'))})
                ut.log_wandb(wandb_metrics)
        else:
            for key in metrics:
                metrics[key].append(0.0)
    
    total_time = time.time() - start_time
    avg_epoch_time = total_time / config['epochs']
    
    if verbose >= 0:
        sim_info = f"K: {config['u_K']}-{config['i_K']} | " if config['edge'] == 'knn' else ""
        if verbose == 2:
            print()  # Add newline after compact progress updates
        print("-" * 220)
        print(f"data: {C.BLUE}{dataset_name}{C.RESET} (u-{C.RED}{data.u_n}{C.RESET}, i-{C.RED}{data.i_n}{C.RESET}, train-{data.train_n}, test-{data.test_n}) | model: {C.GREEN}{config['model']}{C.RESET} | seed: {g_seed} | L: {config['layers']} | {sim_info}batch: {config['batch']} | lr: {config['lr']} | decay: {config['decay']} | margin: {config['margin']} | samples: {config['samples']} | emb: {config['emb_dim']} | n_temp: {config['norm_temp']} | graph: {C.GREEN}{graph_type}{C.RESET}")
        print(f"best_R@{config['eval_K']}: {C.RED}{best_recall:.4f}{C.RESET} | best_P@{config['eval_K']}: {C.RED}{best_prec:.4f}{C.RESET} | best_NDCG@{config['eval_K']}: {C.BLUE}{best_ncdg:.4f}{C.RESET} | best_epoch: {C.RED}{best_epoch}{C.RESET} | time: {total_time/60:.1f}m | avg/epoch: {avg_epoch_time:.1f}s")
        
        # Get final metrics with all K values
        final_results = Test(data, model, config['epochs'], K=config['eval_K'])
        print(f"\nFinal Results:")
        print(f"NDCG@5: {final_results['ndcg@5']:.4f}, NDCG@10: {final_results['ndcg@10']:.4f}, NDCG@20: {final_results['ndcg']:.4f}")
        print(f"HR@5: {final_results['hr@5']:.4f}, HR@10: {final_results['hr@10']:.4f}")
        print(f"Recall@5: {final_results['recall@5']:.4f}, Recall@10: {final_results['recall@10']:.4f}, Recall@20: {final_results['recall']:.4f}")
    
    ut.finish_wandb(run)
    return losses, metrics

