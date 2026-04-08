'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import torch
from torch import nn

class SparseGCN(nn.Module):
    """Unified sparse GCN implementation for baseline models"""
    def __init__(self, emb_dim, l_n, u_n, i_n, **kwargs):
        super().__init__()
        self.u_n = u_n
        self.i_n = i_n
        self.l_n = l_n
        self.emb_dim = emb_dim
        
        self.embedding_user = nn.Embedding(u_n, emb_dim)
        self.embedding_item = nn.Embedding(i_n, emb_dim)
        
        # Initialize with Xavier uniform for better convergence
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
    
    def computer(self, graph):
        """Propagate using pre-computed sparse graph"""
        # Concatenate embeddings once
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        
        # Layer-wise propagation
        for layer in range(self.l_n):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        # Average all layer representations
        light_out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return torch.split(light_out, [self.u_n, self.i_n])
    
    def forward(self, graph):
        """Forward pass using pre-computed graph"""
        # Initial embeddings
        emb0 = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        
        # Get final embeddings
        users, items = self.computer(graph)
        out = torch.cat([users, items])
        
        return emb0, out

class LightGCN(nn.Module):
    """LightGCN: Graph Convolution without Feature Transformation or Non-linear Activation
    
    Core Logic:
    1. Initialize user/item embeddings
    2. Propagate through L layers: E^(l+1) = (D^-0.5 A D^-0.5) E^(l)  
    3. Final embedding: average of all layers E = (1/L+1) * sum(E^(l))
    4. Prediction: dot product of user and item final embeddings
    """
    def __init__(self, emb_dim, l_n, u_n, i_n, **kwargs):
        super().__init__()
        self.u_n = u_n
        self.i_n = i_n
        self.l_n = l_n
        self.emb_dim = emb_dim
        
        self.gcn_model = SparseGCN(emb_dim, l_n, u_n, i_n, **kwargs)

    def forward(self, data):
        """Forward pass using LightGCN graph from data object"""
        return self.gcn_model(data.lightgcn_graph)

    def encode_minibatch(self, u, i_pos, i_neg, data):
        """Encode minibatch using data object"""
        emb0, out = self(data)
        
        if i_neg.dim() == 1:  # samples=1 case
            neg_emb = out[i_neg]
            neg_emb0 = emb0[i_neg]
        else:  # samples>1 case  
            neg_emb = out[i_neg.flatten()].view(i_neg.shape + (-1,))
            neg_emb0 = emb0[i_neg.flatten()].view(i_neg.shape + (-1,))
        
        return (
            out[u], out[i_pos], neg_emb,
            emb0[u], emb0[i_pos], neg_emb0
        )
    
    def predict(self, u, i, data):
        """Predict using data object"""
        emb0, out = self(data)
        
        # For sparse models, out is concatenated [users, items]
        u_embeds = out[u]
        i_embeds = out[self.u_n + i]  # Items start after users
        
        return torch.matmul(u_embeds, i_embeds.t())

class BaselineWrapper(nn.Module):
    """Unified wrapper for baseline models"""
    def __init__(self, emb_dim, l_n, u_n, i_n, model, **kwargs):
        super().__init__()
        
        assert model in ['lightGCN'], f'Baseline model must be lightGCN, got {model}'
        
        self.model = model
        self.u_n = u_n
        self.i_n = i_n
        self.l_n = l_n
        self.emb_dim = emb_dim
        
        # Initialize the appropriate model
        if model == 'lightGCN':
            self.baseline_model = LightGCN(emb_dim, l_n, u_n, i_n, **kwargs)
    
    def forward(self, data):
        """Forward pass using appropriate baseline model"""
        return self.baseline_model(data)
    
    def encode_minibatch(self, u, i_pos, i_neg, data):
        """Encode minibatch using appropriate baseline model"""
        return self.baseline_model.encode_minibatch(u, i_pos, i_neg, data)
    
    def predict(self, u, i, data):
        """Predict using appropriate baseline model"""
        return self.baseline_model.predict(u, i, data)