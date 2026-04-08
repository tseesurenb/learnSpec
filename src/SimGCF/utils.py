'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import random

import numpy as np
import psutil
import torch
import wandb

from config import config, C

def init_wandb():
    if not config['wandb']:
        return None
    
    return wandb.init(
        entity="tseesuren-novelsoft",
        project=config['model'],
        name=f"{config['dataset']}_L{config['layers']}_K{config['u_K']}-{config['i_K']}_S{config['samples']}_E{config['epochs']}",
        config=config,
        mode="online"
    )

def log_wandb(metrics_dict):
    if config['wandb']:
        wandb.log(metrics_dict)

def finish_wandb(run):
    if config['wandb'] and run is not None:
        run.finish()

def print_metrics(recalls, precs, f1s, ndcg, max_indices, stats=None, total_time=None):
    sim_info = f"sim_K: {config['u_K']}-{config['i_K']} | " if config['edge'] == 'knn' else ""
    time_info = f" | time: {total_time/60:.1f}m" if total_time else ""
    print("-" * 120)
    print(f"dataset: {C.BLUE}{config['dataset']}{C.RESET} | model: {C.GREEN}{config['model']}{C.RESET} | layers: {config['layers']} | {sim_info}best_R@20: {C.RED}{np.mean(recalls):.4f}{C.RESET} | best_NDCG: {C.BLUE}{np.mean(ndcg):.4f}{C.RESET} | best_epoch: {C.RED}{max_indices}{C.RESET}{time_info}")
    
def get_metrics(u_emb, i_emb, data, K, batch_size=1024):
    u_emb = u_emb.cpu()
    i_emb = i_emb.cpu()
    u_num = u_emb.shape[0]
    i_num = i_emb.shape[0]
    
    topk_list = []
    for start in range(0, u_num, batch_size):
        end = min(start + batch_size, u_num)
        scores = torch.matmul(u_emb[start:end], i_emb.t())
        
        for idx, u_id in enumerate(range(start, end)):
            if u_id in data.adj_list:
                i_pos = data.adj_list[u_id]['i_pos']
                scores[idx, i_pos] = float('-inf')
        
        topk_list.append(torch.topk(scores, K).indices.detach().cpu())

    topk_indices = torch.cat(topk_list).numpy()
        
    hit_counts = np.zeros(u_num, dtype=np.float32)
    dcg_scores = np.zeros(u_num, dtype=np.float32)
    idcg_scores = np.zeros(u_num, dtype=np.float32)
    
    discount = 1.0 / np.log2(np.arange(2, K + 2))
    
    for u_id in range(u_num):
        test_items = data._test_i_sets[u_id]
        if not test_items:
            continue
            
        pred_items = topk_indices[u_id]
        hit_counts[u_id] = sum(1 for item in pred_items if item in test_items)
        
        relevance = np.array([1.0 if item in test_items else 0.0 for item in pred_items], dtype=np.float32)
        dcg_scores[u_id] = np.sum(relevance * discount)
        
        test_len = len(test_items)
        ideal_len = min(test_len, K)
        if ideal_len > 0:
            idcg_scores[u_id] = np.sum(discount[:ideal_len])
    
    recall = np.where(data._test_lengths > 0, hit_counts / data._test_lengths, 0.0)
    precision = hit_counts / K
    ndcg = np.where(idcg_scores > 0, dcg_scores / idcg_scores, 0.0)
    
    return np.mean(recall), np.mean(precision), np.mean(ndcg)

def get_metrics_multi_k(u_emb, i_emb, data, K_list=[5, 10, 20], batch_size=1024):
    """Calculate metrics for multiple K values"""
    u_emb = u_emb.cpu()
    i_emb = i_emb.cpu()
    u_num = u_emb.shape[0]
    i_num = i_emb.shape[0]
    max_K = max(K_list)
    
    topk_list = []
    for start in range(0, u_num, batch_size):
        end = min(start + batch_size, u_num)
        scores = torch.matmul(u_emb[start:end], i_emb.t())
        
        for idx, u_id in enumerate(range(start, end)):
            if u_id in data.adj_list:
                i_pos = data.adj_list[u_id]['i_pos']
                scores[idx, i_pos] = float('-inf')
        
        topk_list.append(torch.topk(scores, max_K).indices.detach().cpu())

    topk_indices = torch.cat(topk_list).numpy()
    
    results = {}
    for K in K_list:
        hit_counts = np.zeros(u_num, dtype=np.float32)
        dcg_scores = np.zeros(u_num, dtype=np.float32)
        idcg_scores = np.zeros(u_num, dtype=np.float32)
        
        discount = 1.0 / np.log2(np.arange(2, K + 2))
        
        for u_id in range(u_num):
            test_items = data._test_i_sets[u_id]
            if not test_items:
                continue
                
            pred_items = topk_indices[u_id][:K]
            hit_counts[u_id] = sum(1 for item in pred_items if item in test_items)
            
            relevance = np.array([1.0 if item in test_items else 0.0 for item in pred_items], dtype=np.float32)
            dcg_scores[u_id] = np.sum(relevance * discount)
            
            test_len = len(test_items)
            ideal_len = min(test_len, K)
            if ideal_len > 0:
                idcg_scores[u_id] = np.sum(discount[:ideal_len])
        
        valid_users = data._test_lengths > 0
        recall = np.where(valid_users, hit_counts / data._test_lengths, 0.0)
        precision = hit_counts / K
        ndcg = np.where(idcg_scores > 0, dcg_scores / idcg_scores, 0.0)
        hr = np.where(valid_users, (hit_counts > 0).astype(float), 0.0)
        
        results[f'recall@{K}'] = np.mean(recall)
        results[f'precision@{K}'] = np.mean(precision)
        results[f'ndcg@{K}'] = np.mean(ndcg)
        results[f'hr@{K}'] = np.mean(hr)
    
    return results

def get_metrics_itemknn(model, data, K, batch_size=128):
    """Evaluate non-parametric models (ItemKNN/UserKNN/ItemPop) using their predict method instead of embeddings - optimized"""
    print(f"Evaluating non-parametric model with {data.u_n} users and {data.i_n} items...")
    
    u_num = data.u_n
    i_num = data.i_n
    
    topk_list = []
    for start in range(0, u_num, batch_size):
        end = min(start + batch_size, u_num)
        print(f"Processing users {start}-{end-1} of {u_num}")
        
        batch_scores = []
        for u_id in range(start, end):
            # Get scores for this user against all items (using the optimized method)
            # Access the underlying ItemKNN model
            if hasattr(model, 'model_impl') and hasattr(model.model_impl, 'baseline_model'):
                itemknn_model = model.model_impl.baseline_model
            else:
                itemknn_model = model
            user_scores = itemknn_model.predict_user_all_items(u_id, data)
            batch_scores.append(user_scores)
        
        scores = torch.stack(batch_scores)
        
        # Remove positive items from ranking (set to -inf)
        for idx, u_id in enumerate(range(start, end)):
            if u_id in data.adj_list:
                i_pos = data.adj_list[u_id]['i_pos']
                # i_pos contains item-only indices [0, n_items)
                scores[idx, i_pos] = float('-inf')
        
        topk_list.append(torch.topk(scores, K).indices.detach().cpu())

    topk_indices = torch.cat(topk_list).numpy()
    print("Finished computing predictions, calculating metrics...")
        
    hit_counts = np.zeros(u_num, dtype=np.float32)
    dcg_scores = np.zeros(u_num, dtype=np.float32)
    idcg_scores = np.zeros(u_num, dtype=np.float32)
    
    discount = 1.0 / np.log2(np.arange(2, K + 2))
    
    for u_id in range(u_num):
        test_items = data._test_i_sets[u_id]
        if not test_items:
            continue
            
        pred_items = topk_indices[u_id]
        hit_counts[u_id] = sum(1 for item in pred_items if item in test_items)
        
        relevance = np.array([1.0 if item in test_items else 0.0 for item in pred_items], dtype=np.float32)
        dcg_scores[u_id] = np.sum(relevance * discount)
        
        test_len = len(test_items)
        ideal_len = min(test_len, K)
        idcg_scores[u_id] = np.sum(discount[:ideal_len]) if ideal_len > 0 else 0.0
    
    valid_users = np.sum(idcg_scores > 0)
    if valid_users == 0:
        return 0.0, 0.0, 0.0
    
    precision_per_user = hit_counts / K
    recall_per_user = hit_counts / np.array([len(data._test_i_sets[u]) if len(data._test_i_sets[u]) > 0 else 1 for u in range(u_num)])
    ndcg_per_user = dcg_scores / np.maximum(idcg_scores, 1e-8)
    
    precision = np.mean(precision_per_user[idcg_scores > 0])
    recall = np.mean(recall_per_user[idcg_scores > 0]) 
    ndcg = np.mean(ndcg_per_user[idcg_scores > 0])
    
    return recall, precision, ndcg

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def minibatch(*tensors, batch_size):
    for i in range(0, len(tensors[0]), batch_size):
        yield tuple(x[i:i + batch_size] for x in tensors)

def mem_usage(device='cpu'):
    try:
        process = psutil.Process()
        ram_gb = process.memory_info().rss / (1024**3)
        
        if device == 'cuda' and torch.cuda.is_available():
            gpu_gb = torch.cuda.memory_allocated() / (1024**3)
            return f"mem {ram_gb:.1f}R/{gpu_gb:.1f}G"
        else:
            return f"mem {ram_gb:.1f}G"
    except Exception:
        return "mem ?G"
                 
def shuffle(*arrays, **kwargs):
    need_idx = kwargs.get('indices', False)
    
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs must have same length')
    
    idx = np.arange(len(arrays[0]))
    np.random.shuffle(idx)
    
    result = tuple(x[idx] for x in arrays) if len(arrays) > 1 else arrays[0][idx]
    
    return (result, idx) if need_idx else result


