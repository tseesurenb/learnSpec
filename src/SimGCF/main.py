'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import torch
from procedure import run_experiment
from utils import set_seed
from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

set_seed(config['seed'])
losses, metrics = run_experiment(config['dataset'], 1, config['seed'], device, config['verbose'])