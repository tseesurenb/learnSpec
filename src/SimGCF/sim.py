'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from config import C

def _calc_adaptive_k(interaction_counts, avg_count, top_k, min_k, max_k, idx):
    k = int(top_k * np.sqrt(interaction_counts[idx] / avg_count))
    return max(min_k, min(k, max_k))

def _setup_progress_bar(sim_name, adaptive, top_k, n_rows):
    desc = f'Preparing {C.RED}{"adaptive " if adaptive else ""}{sim_name}{C.RESET} similarity' + (f' | top K: {top_k}' if not adaptive else '')
    return tqdm(range(n_rows), desc=desc, bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")

def _filter_top_k(sim_matrix, top_k, adaptive, interaction_counts, avg_count, min_k, max_k, verbose, sim_name):
    data, rows, cols = [], [], []
    k_vals = [] if adaptive else None
    
    pbar = _setup_progress_bar(sim_name, adaptive, top_k, sim_matrix.shape[0])
    
    for i in pbar:
        k = _calc_adaptive_k(interaction_counts, avg_count, top_k, min_k, max_k, i) if adaptive else top_k
        
        if adaptive:
            k_vals.append(k)
            if verbose > 1 and i % 1000 == 0:
                pbar.set_postfix_str(f"User {i}: {interaction_counts[i]} interactions → K={k}")
        
        # Get row data (sparse or dense)
        if hasattr(sim_matrix, 'getrow'):  # Sparse
            row = sim_matrix.getrow(i).tocoo()
            if row.nnz == 0:
                continue
            row_data, row_idx = row.data, row.col
        else:  # Dense
            row_data = sim_matrix[i]
            if np.count_nonzero(row_data) == 0:
                continue
            
            if len(row_data) > k:
                top_idx = np.argpartition(-row_data, k-1)[:k]
                top_idx = top_idx[np.argsort(-row_data[top_idx])]
            else:
                top_idx = np.argsort(-row_data)
                
            data.extend(row_data[top_idx])
            rows.extend([i] * len(top_idx))
            cols.extend(top_idx)
            continue
        
        # For sparse: select top-k
        if row_data.size > k:
            top_idx = np.argpartition(-row_data, k-1)[:k]
            top_idx = top_idx[np.argsort(-row_data[top_idx])]
        else:
            top_idx = np.argsort(-row_data)
        data.extend(row_data[top_idx])
        rows.extend([i] * len(top_idx))
        cols.extend(row_idx[top_idx])
    
    return data, rows, cols, k_vals

def jaccard(matrix, top_k=20, self_loop=False, verbose=-1, adaptive=False, min_k=10, max_k=100):
    if verbose > 0:
        print('Computing Jaccard similarity...')
    
    # Handle both sparse and dense input matrices efficiently
    if hasattr(matrix, 'toarray'):  # Sparse matrix
        bin_matrix = (matrix > 0).astype(int)
        if not isinstance(bin_matrix, csr_matrix):
            bin_matrix = bin_matrix.tocsr()
    else:  # Dense matrix
        bin_matrix = csr_matrix((matrix > 0).astype(int))
    
    if adaptive:
        counts = np.array(bin_matrix.sum(axis=1)).flatten()
        avg_count = max(1, np.mean(counts))
        if verbose > 0:
            print(f'Adaptive K: base={top_k}, range=[{min_k}, {max_k}]')
    else:
        counts = avg_count = None
    
    # Pre-compute row sums for efficiency
    row_sums = np.array(bin_matrix.sum(axis=1)).flatten()
    
    # Vectorized Jaccard computation using chunked processing for memory efficiency
    chunk_size = min(500, bin_matrix.shape[0])  # Process in chunks to manage memory
    data, rows, cols = [], [], []
    k_vals = [] if adaptive else None
    
    pbar = _setup_progress_bar('jaccard', adaptive, top_k, bin_matrix.shape[0])
    
    for start in range(0, bin_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, bin_matrix.shape[0])
        chunk = bin_matrix[start:end]
        
        # Vectorized intersection computation for chunk
        intersection = chunk.dot(bin_matrix.T)
        if hasattr(intersection, 'toarray'):
            intersection = intersection.toarray()
        
        # Vectorized union computation
        union = row_sums[start:end, np.newaxis] + row_sums[np.newaxis, :] - intersection
        
        # Vectorized Jaccard similarity computation
        with np.errstate(divide='ignore', invalid='ignore'):
            sim_chunk = np.divide(intersection.astype(np.float32), 
                                union.astype(np.float32), 
                                out=np.zeros_like(intersection, dtype=np.float32), 
                                where=union != 0)
        
        # Process each row in the chunk
        for i in range(sim_chunk.shape[0]):
            global_i = start + i
            pbar.update(1)
            
            k = _calc_adaptive_k(counts, avg_count, top_k, min_k, max_k, global_i) if adaptive else top_k
            
            if adaptive:
                k_vals.append(k)
                if verbose > 1 and global_i % 1000 == 0:
                    pbar.set_postfix_str(f"Row {global_i}: {row_sums[global_i]} items → K={k}")
            
            sim_i = sim_chunk[i]
            
            # Handle self-loop
            if self_loop:
                sim_i[global_i] = 1.0
            else:
                sim_i[global_i] = 0.0
            
            # Skip if no similarities
            if np.count_nonzero(sim_i) == 0:
                continue
                
            # Get top-k similarities efficiently
            if len(sim_i) > k and np.count_nonzero(sim_i) > k:
                # Use argpartition for better performance with large k
                top_idx = np.argpartition(-sim_i, k-1)[:k]
                top_idx = top_idx[np.argsort(-sim_i[top_idx])]
            else:
                # For small arrays or when we need all non-zero values
                nonzero_idx = np.nonzero(sim_i)[0]
                if len(nonzero_idx) == 0:
                    continue
                top_idx = nonzero_idx[np.argsort(-sim_i[nonzero_idx])]
            
            # Only keep non-zero similarities
            mask = sim_i[top_idx] > 0
            if not np.any(mask):
                continue
                
            top_idx = top_idx[mask]
            data.extend(sim_i[top_idx])
            rows.extend([global_i] * len(top_idx))
            cols.extend(top_idx)
    
    pbar.close()
    
    result = coo_matrix((data, (rows, cols)), shape=bin_matrix.shape)
    
    if adaptive and k_vals:
        k_arr = np.array(k_vals)
        return result.tocsr(), k_arr.mean()
    
    return result.tocsr(), None


def cosine(matrix, top_k=20, self_loop=False, verbose=-1, adaptive=True, min_k=10, max_k=100):
    sparse_matrix = csr_matrix(matrix)
    sparse_matrix.data = (sparse_matrix.data > 0).astype(int)
    
    if adaptive:
        counts = np.array(sparse_matrix.sum(axis=1)).flatten()
        avg_count = max(1, np.mean(counts))
    else:
        counts = avg_count = None

    sim_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    sim_matrix.setdiag(1 if self_loop else 0)
    
    data, rows, cols, k_vals = _filter_top_k(sim_matrix, top_k, adaptive, counts, avg_count, min_k, max_k, verbose, 'cosine')
    
    result = coo_matrix((data, (rows, cols)), shape=sim_matrix.shape)
    
    if adaptive and k_vals:
        k_arr = np.array(k_vals)
        return result.tocsr(), k_arr.mean()
    
    return result.tocsr(), None


