import argparse
import torch

SPLIT_RATIO = 0.7
SPLIT_SEED = 42

INIT_TYPES = ['uniform', 'lowpass', 'highpass', 'bandpass', 'butterworth', 'bandreject', 'decay', 'rise', 'plateau']
DATASETS = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
OPTIMIZERS = ['rmsprop', 'adam']
POLYNOMIAL_BASIS = ['bernstein', 'cheby', 'direct', 'adaptive']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=DATASETS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view', type=str, default='ui', help='u=user, i=item, ui=both')
    parser.add_argument('--u_eigen', type=int, default=25)
    parser.add_argument('--i_eigen', type=int, default=130)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--f_order', type=int, default=32)
    parser.add_argument('--f_init', type=str, default='bandpass', choices=INIT_TYPES)
    parser.add_argument('--f_poly', type=str, default='bernstein', choices=POLYNOMIAL_BASIS)
    parser.add_argument('--f_dropout', type=float, default=0.0)
    parser.add_argument('--f_act', type=str, default='sigmoid', choices=['sigmoid', 'softplus', 'tanh', 'none'])
    parser.add_argument('--opt', type=str, default='rmsprop', choices=OPTIMIZERS)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--ensemble', action='store_true', default=False, help='Use ensemble of multiple filters')
    parser.add_argument('--n_filters', type=int, default=3, help='Number of subfilters for ensemble')
    parser.add_argument('--smooth', type=float, default=0.01, help='Smoothness regularization weight')
    parser.add_argument('--guf', action='store_true', default=False, help='Group-adaptive spectral filter')
    parser.add_argument('--n_groups', type=int, default=5, help='Number of user groups for --guf')
    parser.add_argument('--cross', action='store_true', default=False, help='Cross-view spectral interaction')
    parser.add_argument('--svd', action='store_true', default=False, help='SVD view: direct spectral filtering on singular values')
    parser.add_argument('--svd_k', type=int, default=50, help='Number of singular values for SVD view')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train/val split ratio for sub-eigenspace learning')
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
        'f_dropout': args.f_dropout, 'f_act': args.f_act,
        'opt': args.opt, 'lr': args.lr, 'decay': args.decay,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'patience': args.patience, 'eval_every': args.eval_every,
        'guf': args.guf, 'n_groups': args.n_groups, 'cross': args.cross,
        'ensemble': args.ensemble, 'n_filters': args.n_filters, 'smooth': args.smooth,
        'svd': args.svd, 'svd_k': args.svd_k,
        'split_ratio': args.split_ratio,
        'infer': args.infer, 'save': args.save,
        'device': device, 'topks': [20],
    }
