'''
Created on Sep 1, 2024
Pytorch Implementation of SimGCF: A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
Combined configuration and argument parsing module
'''

import argparse

# ===== Dataset Configuration =====
SUPPORTED_DATASETS = ['amazon-book', 'yelp2018', 'gowalla', 'ml-100k', 'ml-1m']

# ===== ANSI Colors (centralized) =====
class Colors:
    RED = "\033[1;31m"      # br - bold red
    BLUE = "\033[1;34m"     # bb - bold blue  
    GREEN = "\033[1;32m"    # bg - bold green
    BOLD = "\033[1m"        # b - bold
    RESET = "\033[0m"       # rs - reset

# Global color instance for easy import
C = Colors()

def parse_args():
    parser = argparse.ArgumentParser(prog="SimGCF", description="Dynamic GCN-based CF recommender")
    
    # ===== Basic Configuration =====
    parser.add_argument('--model', type=str, default='SimGCF', help='rec-model, support [LightGCN, SimGCF]')
    parser.add_argument('--dataset', type=str, default='ml-100k', help="available datasets: [ml-100k, ml-1m, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--verbose', type=int, default=-1)
    
    # ===== Model Architecture =====
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of GCN")
    parser.add_argument('--norm_temp', type=float, default=0.1, help="temperature for symmetric softmax normalization for neighbor aggregation")
    
    # ===== Training Configuration =====
    parser.add_argument('--epochs', type=int, default=51)
    parser.add_argument('--eval_freq', type=int, default=1, help="evaluate model every N epochs")
    parser.add_argument('--batch', type=int, default=2048, help="the batch size for bpr loss training procedure")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--min_lr', type=float, default=0.002, help="minimum learning rate for schedulers")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for L2 regularization")
    parser.add_argument('--margin', type=float, default=0.0, help="the margin in BPR loss")
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--eval_K', type=int, default=20, help="@k test list")
    
    # ===== Learning Rate Scheduling =====
    parser.add_argument('--lr_sched', type=str, default='none', help="LR scheduler: 'step', 'plateau', or 'none'")
    parser.add_argument('--lr_factor', type=float, default=0.8, help="factor to reduce LR by (for step/plateau scheduler)")
    parser.add_argument('--lr_step', type=int, default=2, help="step size for StepLR scheduler")
    parser.add_argument('--lr_patience', type=int, default=1, help="patience for ReduceLROnPlateau scheduler")
    
    # ===== Similarity & Graph Configuration =====
    parser.add_argument('--sim', type=str, default='cos', help='similarity metric for both users and items: cos (cosine) or jac (jaccard)')
    parser.add_argument('--edge', type=str, default='knn', help='options are knn (similarity graph) and bi (bi-partite graph)')
    parser.add_argument('--u_K', type=int, default=80)
    parser.add_argument('--i_K', type=int, default=10)
    
    
    # ===== Experiment & Debug =====
    parser.add_argument('--wandb', action='store_true', help="use Weights & Biases for experiment tracking")

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Create config dictionary (organized by category)
config = {
    # Basic Configuration
    'model': args.model,
    'dataset': args.dataset,
    'seed': args.seed,
    'verbose': args.verbose,
    
    # Model Architecture
    'emb_dim': args.emb_dim,
    'layers': args.layers,
    'norm_temp': args.norm_temp,
    
    # Training Configuration
    'epochs': args.epochs,
    'eval_freq': args.eval_freq,
    'batch': args.batch,
    'lr': args.lr,
    'min_lr': args.min_lr,
    'decay': args.decay,
    'margin': args.margin,
    'samples': args.samples,
    'eval_K': args.eval_K,
    
    # Learning Rate Scheduling
    'lr_sched': args.lr_sched,
    'lr_factor': args.lr_factor,
    'lr_step': args.lr_step,
    'lr_patience': args.lr_patience,
    
    # Similarity & Graph Configuration
    'sim': args.sim,
    'edge': args.edge,
    'u_K': args.u_K,
    'i_K': args.i_K,
    
    
    # Experiment & Debug
    'wandb': args.wandb,
}