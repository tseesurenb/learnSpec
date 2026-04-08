'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import torch
import numpy as np
import polars as pl
import random
import os
from config import config
import sim
from scipy.sparse import coo_matrix
import scipy.sparse as sp

class Data:
    def __init__(self, dataset_name, device):
        self.dataset_name = dataset_name
        self.device = device
        self._similarity_cache = {}  # Cache for similarity matrices
        self._load(dataset_name)
        self._build_graphs()
    
    def _load(self, dataset_name):
        train_file = f'../../data/{dataset_name}/train.txt'
        test_file = f'../../data/{dataset_name}/test.txt'
        
        # Error handling for missing files
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Dataset files not found for {dataset_name}")
        
        # Direct parsing to avoid intermediate lists
        train_pairs, test_pairs = [], []
        
        # Parse train file efficiently
        with open(train_file, 'r') as f:
            for line in f:
                items = [int(x) for x in line.strip().split()]
                user_id = items[0]
                train_pairs.extend([(user_id, item_id) for item_id in items[1:]])
        
        # Parse test file efficiently  
        with open(test_file, 'r') as f:
            for line in f:
                items = [int(x) for x in line.strip().split()]
                user_id = items[0]
                test_pairs.extend([(user_id, item_id) for item_id in items[1:]])
        
        # Create DataFrames directly
        train_df = pl.DataFrame(train_pairs, schema=['user_id', 'item_id'], orient="row")
        test_df = pl.DataFrame(test_pairs, schema=['user_id', 'item_id'], orient="row")
        
        # Build adjacency list with consistent np.array types
        self.adj_list = {}
        all_train_items = set()
        for u_id, items in train_df.group_by('user_id').agg(pl.col('item_id')).iter_rows():
            items_array = np.array(items, dtype=np.int32)
            self.adj_list[u_id] = {'i_pos': items_array}
            all_train_items.update(items)
        
        # Build test dict with consistent np.array types  
        self.test_dict = {}
        all_test_items = set()
        for u_id, items in test_df.group_by('user_id').agg(pl.col('item_id')).iter_rows():
            self.test_dict[u_id] = np.array(items, dtype=np.int32)  # Consistent with adj_list
            all_test_items.update(items)
        
        # Memory cleanup
        del train_pairs, test_pairs, train_df, test_df
        
        self._finalize_loading(all_train_items, all_test_items)
    
    
    def _finalize_loading(self, all_train_items, all_test_items):
        self.u_n = len(self.adj_list)
        self.i_n = max(max(all_train_items), max(all_test_items) if all_test_items else 0) + 1
        self.train_n = sum(len(data['i_pos']) for data in self.adj_list.values())
        # Fix: handle np.array lengths correctly
        self.test_n = sum(len(items) if hasattr(items, '__len__') else 0 for items in self.test_dict.values())
        
        # Store negative item arrays for simple sampling
        for u_id in self.adj_list:
            pos_set = set(self.adj_list[u_id]['i_pos'])
            self.adj_list[u_id]['i_neg'] = np.array(list(set(range(self.i_n)) - pos_set), dtype=np.int32)
        
        # Pre-allocate training arrays
        self.train_u = np.empty(self.train_n, dtype=np.int32)
        self.train_i_pos = np.empty(self.train_n, dtype=np.int32)
        
        self._u_pos_slices = []
        self._u_neg_lists = []
        idx = 0
        for u_id in sorted(self.adj_list.keys()):
            i_pos = self.adj_list[u_id]['i_pos']
            i_neg = self.adj_list[u_id]['i_neg']
            pos_n = len(i_pos)
            
            if pos_n > 0:
                self.train_u[idx:idx+pos_n] = u_id
                self.train_i_pos[idx:idx+pos_n] = i_pos + self.u_n
                self._u_pos_slices.append((idx, idx+pos_n))
                self._u_neg_lists.append(i_neg)
                idx += pos_n
        
        # Optimize test data structures - avoid redundant conversions
        self._test_i_sets = []
        test_lengths = []
        for u_id in range(self.u_n):
            test_items = self.test_dict.get(u_id, np.array([], dtype=np.int32))
            test_set = set(test_items) if len(test_items) > 0 else set()
            self._test_i_sets.append(test_set)
            test_lengths.append(len(test_items))
        
        self._test_lengths = np.array(test_lengths, dtype=np.float32)
        
        # Create torch tensors
        self.train_u_t = torch.tensor(self.train_u, device=self.device, dtype=torch.long)
        self.train_i_pos_t = torch.tensor(self.train_i_pos, device=self.device, dtype=torch.long)
    
        
    def _build_graphs(self):
        # More memory-efficient edge building using existing training arrays
        u_edges = torch.tensor(self.train_u, dtype=torch.long, device=self.device)
        i_edges = torch.tensor(self.train_i_pos, dtype=torch.long, device=self.device)
        
        # Create bidirectional edges efficiently
        self.bipartite_edge_index = torch.stack((
            torch.cat([u_edges, i_edges]),
            torch.cat([i_edges, u_edges])
        ))
        
        if config['edge'] == 'knn':
            self._build_similarity_graph()
        else:
            self.knn_edge_index = None
            self.knn_edge_attrs = None
    
    def _build_similarity_graph(self):
        # More efficient - avoid intermediate variables
        items_original = self.train_i_pos - self.u_n  # Remove offset
        df = pl.DataFrame({'user_id': self.train_u, 'item_id': items_original})
        knn_edge_index, knn_edge_attrs = self._create_adjmat(df, verbose=config.get('verbose', -1))
        self.knn_edge_index = torch.tensor(knn_edge_index, device=self.device, dtype=torch.long)
        self.knn_edge_attrs = torch.tensor(knn_edge_attrs, device=self.device)
    
    def _create_adjmat(self, df, verbose=-1):
        
        u_ids = df['user_id'].to_numpy()
        i_ids = df['item_id'].to_numpy()
        # Keep as sparse matrix - avoid expensive .toarray() conversion
        ui_matrix = coo_matrix((np.ones(len(df)), (u_ids, i_ids)))
        
        # More efficient cache key - avoid expensive ui_matrix computations
        cache_key = (config['sim'], config['u_K'], config['i_K'], len(u_ids), max(u_ids) if len(u_ids) > 0 else 0, max(i_ids) if len(i_ids) > 0 else 0)
        
        if cache_key in self._similarity_cache:
            if verbose > 0:
                print(f"Using cached {config['sim']} similarity matrices")
            u_sim, i_sim, u_k_mean, i_k_mean = self._similarity_cache[cache_key]
        else:
            if config['sim'] == 'cos':
                u_sim, u_k_mean = sim.cosine(ui_matrix, top_k=config['u_K'], verbose=verbose)
                i_sim, i_k_mean = sim.cosine(ui_matrix.T, top_k=config['i_K'], verbose=verbose)
            elif config['sim'] == 'jac':
                u_sim, u_k_mean = sim.jaccard(ui_matrix, top_k=config['u_K'], verbose=verbose)
                i_sim, i_k_mean = sim.jaccard(ui_matrix.T, top_k=config['i_K'], verbose=verbose)
            else:
                raise ValueError(f'Similarity metric {config["sim"]} not supported. Use "cos" or "jac".')
            
            # Cache the results
            self._similarity_cache[cache_key] = (u_sim, i_sim, u_k_mean, i_k_mean)
        
        if verbose > 0 and u_k_mean is not None and i_k_mean is not None:
            print(f"Computing {config['sim']} similarity: u-K{config['u_K']}→{u_k_mean:.1f}, i-K{config['i_K']}→{i_k_mean:.1f}")
        
        del ui_matrix
            
        # More efficient COO handling
        u_sim_coo = u_sim.tocoo() if not hasattr(u_sim, 'row') else u_sim
        i_sim_coo = i_sim.tocoo() if not hasattr(i_sim, 'row') else i_sim
        
        u_edge_index = np.vstack((u_sim_coo.row, u_sim_coo.col))
        u_edge_attrs = u_sim_coo.data.astype(np.float32)
        
        i_edge_index = np.vstack((i_sim_coo.row, i_sim_coo.col))
        i_edge_attrs = i_sim_coo.data.astype(np.float32)
        
        # Compute dimensions and offset inline
        n_u = u_sim.shape[0]
        i_edge_index_offset = i_edge_index + n_u
        
        # Simplified validation (only check if edges exist)
        if i_edge_index_offset.size > 0 and np.max(i_edge_index_offset) >= n_u + i_sim.shape[0]:
            raise ValueError("Item similarity offset creates out-of-bounds indices")
        if u_edge_index.size > 0 and np.max(u_edge_index) >= n_u:
            raise ValueError("User similarity ID out of bounds")
        
        del u_sim, i_sim
        
        # Pre-allocate arrays to avoid hstack memory copies
        total_edges = u_edge_index.shape[1] + i_edge_index_offset.shape[1]
        edge_index = np.empty((2, total_edges), dtype=u_edge_index.dtype)
        edge_attrs = np.empty(total_edges, dtype=np.float32)
        
        # Copy data efficiently
        u_edges = u_edge_index.shape[1]
        edge_index[:, :u_edges] = u_edge_index
        edge_index[:, u_edges:] = i_edge_index_offset
        edge_attrs[:u_edges] = u_edge_attrs
        edge_attrs[u_edges:] = i_edge_attrs
        
        return edge_index, edge_attrs
    
    
    def sample_negatives(self, epoch=None, seed=42):
        if seed is not None:
            np.random.seed(seed + (epoch or 0))
        
        if config['samples'] == 1:
            i_neg = np.empty(self.train_n, dtype=np.int32)
        else:
            i_neg = np.empty((self.train_n, config['samples']), dtype=np.int32)
        
        for (start, end), neg_list in zip(self._u_pos_slices, self._u_neg_lists):
            pos_n = end - start
            
            if config['samples'] == 1:
                i_neg[start:end] = np.random.choice(neg_list, size=pos_n, replace=True) + self.u_n
            else:
                i_neg[start:end] = np.random.choice(neg_list, size=(pos_n, config['samples']), replace=True) + self.u_n
        
        return i_neg
    
    def create_lightgcn_sparse_matrix(self):
        """Create normalized sparse adjacency matrix for LightGCN"""
        n_nodes = self.u_n + self.i_n
        edge_index = self.bipartite_edge_index
        
        # Compute degree directly from edge_index
        from_, to_ = edge_index
        degree = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)
        degree.scatter_add_(0, from_, torch.ones_like(from_, dtype=torch.float32))
        degree.scatter_add_(0, to_, torch.ones_like(to_, dtype=torch.float32))
        
        # Compute normalization: D^(-1/2)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        
        # Apply normalization to edges: norm[i,j] = 1/sqrt(deg[i] * deg[j])
        edge_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        return torch.sparse_coo_tensor(edge_index, edge_norm, (n_nodes, n_nodes), 
                                     dtype=torch.float32, device=self.device).coalesce()
    
    def create_simgcf_sparse_matrix(self):
        """Create sparse matrix for SimGCF with similarity-based normalization"""
        if config['edge'] == 'knn':
            edge_index = self.knn_edge_index
            edge_attrs = self.knn_edge_attrs
        else:
            edge_index = self.bipartite_edge_index
            edge_attrs = torch.ones(self.bipartite_edge_index.shape[1], device=self.device)
        
        n_nodes = self.u_n + self.i_n
        
        # Compute similarity-based normalization
        norm_weights = self._compute_similarity_normalization(edge_index, edge_attrs, n_nodes)
        
        return torch.sparse_coo_tensor(edge_index, norm_weights, (n_nodes, n_nodes), 
                                     dtype=torch.float32, device=self.device).coalesce()
    
    def _compute_similarity_normalization(self, edge_index, edge_attrs, num_nodes):
        """Compute similarity-based normalization weights - memory optimized"""
        temp = config['norm_temp']
        if temp == 0:
            return torch.ones_like(edge_attrs)
        
        device = edge_index.device
        from_, to_ = edge_index
        
        # Compute exp without modifying original edge_attrs
        exp_attrs = torch.exp(edge_attrs / temp)
        
        # Compute softmax normalization sums
        in_sum = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        out_sum = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        
        in_sum.scatter_add_(0, to_, exp_attrs)
        out_sum.scatter_add_(0, from_, exp_attrs)
        
        # Add epsilon for numerical stability
        in_sum.add_(1e-16)
        out_sum.add_(1e-16)
        
        # Compute symmetric normalization: sqrt((exp_attrs / in_sum[to_]) * (exp_attrs / out_sum[from_]))
        # = exp_attrs / sqrt(in_sum[to_] * out_sum[from_])
        normalization_factor = torch.sqrt(in_sum[to_] * out_sum[from_])
        
        return exp_attrs / normalization_factor
    
    @property
    def lightgcn_graph(self):
        """Get or create LightGCN sparse matrix"""
        if not hasattr(self, '_lightgcn_graph'):
            self._lightgcn_graph = self.create_lightgcn_sparse_matrix()
        return self._lightgcn_graph
    
    @property
    def simgcf_graph(self):
        """Get or create SimGCF sparse matrix"""
        if not hasattr(self, '_simgcf_graph'):
            self._simgcf_graph = self.create_simgcf_sparse_matrix()
        return self._simgcf_graph
    
    @property
    def identity_matrix(self):
        """Get or create identity matrix for NGCF bi-interaction"""
        if not hasattr(self, '_identity_matrix'):
            n_nodes = self.u_n + self.i_n
            # Create sparse identity matrix
            indices = torch.arange(n_nodes, device=self.device)
            values = torch.ones(n_nodes, device=self.device)
            self._identity_matrix = torch.sparse_coo_tensor(
                torch.stack([indices, indices]), 
                values, 
                (n_nodes, n_nodes),
                device=self.device
            ).coalesce()
        return self._identity_matrix
    
    
    def __repr__(self):
        return f"Data(dataset={self.dataset_name}, users={self.u_n}, items={self.i_n}, train={self.train_n}, test={self.test_n})"