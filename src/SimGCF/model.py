'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import torch
from torch import nn
from config import config
from baselines import BaselineWrapper

class SimGCF(nn.Module):
    """SimGCF: Dynamic Similarity-Centric Graph Convolutional Network"""
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
        """Propagate using pre-computed SimGCF graph with similarity integration"""
        # Concatenate embeddings once
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        
        # Layer-wise propagation with dynamic similarity
        for layer in range(self.l_n):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        # Average all layer representations
        dysim_out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return torch.split(dysim_out, [self.u_n, self.i_n])
    
    def forward(self, data):
        """Forward pass using SimGCF graph from data object"""
        # Initial embeddings
        emb0 = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        
        # Get final embeddings using SimGCF graph
        users, items = self.computer(data.simgcf_graph)
        out = torch.cat([users, items])
        
        return emb0, out

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
        
        # For SimGCF, out is concatenated [users, items]
        u_embeds = out[u]
        i_embeds = out[self.u_n + i]  # Items start after users
        
        return torch.matmul(u_embeds, i_embeds.t())

class RecSysGNN(nn.Module):
    """Unified model wrapper for SimGCF and baseline models"""
    def __init__(self, emb_dim, l_n, u_n, i_n, model, **kwargs):
        super().__init__()
        
        assert model in ['lightGCN', 'SimGCF'], f'Model must be lightGCN or SimGCF, got {model}'
        
        self.model = model
        self.u_n = u_n
        self.i_n = i_n
        self.l_n = l_n
        self.emb_dim = emb_dim
        
        # Initialize appropriate model
        if model == 'SimGCF':
            self.model_impl = SimGCF(emb_dim, l_n, u_n, i_n, **kwargs)
        else:
            # Use baseline wrapper for other models
            self.model_impl = BaselineWrapper(emb_dim, l_n, u_n, i_n, model, **kwargs)

    def forward(self, data):
        """Forward pass using appropriate model implementation"""
        return self.model_impl(data)

    def encode_minibatch(self, u, i_pos, i_neg, data):
        """Encode minibatch using appropriate model implementation"""
        return self.model_impl.encode_minibatch(u, i_pos, i_neg, data)
    
    def predict(self, u, i, data):
        """Predict using appropriate model implementation"""
        return self.model_impl.predict(u, i, data)