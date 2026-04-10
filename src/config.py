import argparse
import torch

SPLIT_RATIO = 0.7
SPLIT_SEED = 42

INIT_TYPES = ['uniform', 'lowpass', 'highpass', 'bandpass', 'butterworth', 'decay', 'rise']
DATASETS = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
OPTIMIZERS = ['rmsprop', 'adam']
POLYNOMIAL_BASIS = ['bernstein', 'cheby']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=DATASETS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view', type=str, default='ui', help='u=user, i=item, ui=both')
    parser.add_argument('--u_eigen', type=int, default=20)
    parser.add_argument('--i_eigen', type=int, default=50)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--f_order', type=int, default=32)
    parser.add_argument('--f_init', type=str, default='lowpass', choices=INIT_TYPES)
    parser.add_argument('--f_poly', type=str, default='bernstein', choices=POLYNOMIAL_BASIS)
    parser.add_argument('--f_drop', type=float, default=0.0, help='Spectral dropout: probability of masking eigencomponents during training')
    parser.add_argument('--f_act', type=str, default='sigmoid', choices=['sigmoid', 'softplus'])
    parser.add_argument('--local_fourier', type=int, default=0, help='Local Fourier refinement terms on top of polynomial (0=off)')
    parser.add_argument('--local_rbf', type=int, default=0, help='Local RBF bumps on top of polynomial (0=off)')
    parser.add_argument('--opt', type=str, default='rmsprop', choices=OPTIMIZERS)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False, help='Log detailed filter state at each eval step')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train/val split ratio for sub-eigenspace learning')
    parser.add_argument('--f_reg', type=float, default=0.0, help='Frequency-aware smoothness regularization weight')
    parser.add_argument('--mse_weight', type=float, default=0.0, help='MSE loss weight combined with BPR (0=BPR only)')
    parser.add_argument('--no_bpr', action='store_true', default=False, help='Disable BPR loss, use MSE only')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    return parser.parse_args()


def get_config(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    return {
        'dataset': args.dataset, 'seed': args.seed, 'view': args.view,
        'u_eigen': args.u_eigen, 'i_eigen': args.i_eigen, 'beta': args.beta,
        'f_order': args.f_order, 'f_init': args.f_init, 'poly': args.f_poly,
        'f_drop': args.f_drop, 'f_act': args.f_act, 'f_jitter': args.local_fourier, 'f_rbf': args.local_rbf,
        'opt': args.opt, 'lr': args.lr, 'decay': args.decay,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'patience': args.patience, 'eval_every': args.eval_every,
        'split_ratio': args.split_ratio,
        'f_reg': args.f_reg, 'mse_weight': args.mse_weight, 'no_bpr': args.no_bpr,
        'infer': args.infer, 'save': args.save, 'log': args.log,
        'device': device, 'topks': [20],
    }
