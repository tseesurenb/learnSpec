import argparse
import torch

SPLIT_RATIO = 0.7
SPLIT_SEED = 42

INIT_TYPES = ['uniform', 'lowpass', 'highpass', 'bandpass', 'butterworth', 'decay', 'rise', 'random']
DATASETS = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
OPTIMIZERS = ['rmsprop', 'adam']
POLYNOMIAL_BASIS = ['bernstein', 'cheby', 'direct']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=DATASETS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view', type=str, default='ui', help='u=user, i=item, ui=both')
    parser.add_argument('--u_eigen', type=int, default=20)
    parser.add_argument('--i_eigen', type=int, default=50)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--f_order', type=int, default=32)
    parser.add_argument('--f_init', type=str, default='lowpass', choices=INIT_TYPES)
    parser.add_argument('--f_poly', type=str, default='direct', choices=POLYNOMIAL_BASIS)
    parser.add_argument('--f_drop', type=float, default=0.0, help='Spectral dropout: probability of masking eigencomponents during training')
    parser.add_argument('--f_act', type=str, default='sigmoid', choices=['sigmoid', 'softplus'])
    parser.add_argument('--opt', type=str, default='adam', choices=OPTIMIZERS)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=20, help='Evaluate on validation set every N epochs')
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False, help='Log detailed filter state at each eval step')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train/val split ratio for sub-eigenspace learning')
    parser.add_argument('--f_reg', type=float, default=0.0, help='Frequency-aware smoothness regularization weight')
    parser.add_argument('--loss', type=str, default='bpr', choices=['bpr', 'mse'], help='Loss function: bpr or mse')
    parser.add_argument('--quiet', type=int, default=1, choices=[0, 1], help='0=verbose, 1=progress bar + final result only')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    return parser.parse_args()


def get_config(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and args.quiet == 0:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif device.type == 'cpu' and args.quiet == 0:
        print("Using CPU")

    return {
        'dataset': args.dataset, 'seed': args.seed, 'view': args.view,
        'u_eigen': args.u_eigen, 'i_eigen': args.i_eigen, 'beta': args.beta,
        'f_order': args.f_order, 'f_init': args.f_init, 'poly': args.f_poly,
        'f_drop': args.f_drop, 'f_act': args.f_act,
        'opt': args.opt, 'lr': args.lr, 'decay': args.decay,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'patience': args.patience, 'eval_every': args.eval_every,
        'split_ratio': args.split_ratio,
        'f_reg': args.f_reg, 'loss': args.loss,
        'infer': args.infer, 'save': args.save, 'log': args.log, 'quiet': args.quiet,
        'device': device, 'topks': [20],
    }
