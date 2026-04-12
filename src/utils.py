import os
import json
import pickle
import torch
import numpy as np
from datetime import datetime


class C:
    G = '\033[32;1m'
    B = '\033[94m'
    R = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def count_parameters(model):
    total_params = 0
    param_details = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total_params += n
            param_details[name] = {
                'count': n, 'shape': tuple(param.shape),
                'min': param.data.min().item(), 'max': param.data.max().item(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item() if n > 1 else 0.0
            }
    return total_params, param_details


def get_parameter_changes(model, prev_params):
    changes = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in prev_params:
            diff = (param.data - prev_params[name]).abs()
            changes[name] = {
                'max_change': diff.max().item(),
                'mean_change': diff.mean().item(),
                'relative_change': (diff.mean() / (prev_params[name].abs().mean() + 1e-8)).item()
            }
    return changes


def get_current_parameters(model):
    return {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n[recall_n == 0] = 1
    return {'recall': np.sum(right_pred / recall_n), 'precision': np.sum(right_pred) / k}


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        test_matrix[i, :min(k, len(items))] = 1
    discount = 1. / np.log2(np.arange(2, k + 2))
    idcg = np.sum(test_matrix * discount, axis=1)
    dcg = np.sum(pred_data * discount, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        pred = np.array([x in groundTrue for x in pred_data[i]]).astype('float')
        r.append(pred)
    return np.array(r).astype('float')


def deep_copy_state_dict(state_dict):
    if state_dict is None:
        return None
    return {key: value.clone().detach() for key, value in state_dict.items()}


def create_temp_dataset(validation_data, dataset, partial_train):
    return type('obj', (object,), {
        'valDict': validation_data,
        'n_users': dataset.n_users,
        'm_items': dataset.m_items,
        'allPos': partial_train,
        'getUserPosItems': lambda self, users: [partial_train.get(u, []) for u in users]
    })()


def split_training_data(dataset, split_ratio=0.70, seed=42):
    np.random.seed(seed)
    partial_train = {}
    validation = {}
    for user_id in range(dataset.n_users):
        user_items = list(dataset.allPos[user_id])
        if len(user_items) > 1:
            shuffled = user_items.copy()
            np.random.shuffle(shuffled)
            split_point = max(1, int(len(shuffled) * split_ratio))
            partial_train[user_id] = shuffled[:split_point]
            validation[user_id] = shuffled[split_point:]
        else:
            partial_train[user_id] = user_items
            validation[user_id] = []
    return partial_train, validation


def create_partial_adj_matrix(partial_train, n_users, n_items):
    import scipy.sparse as sp
    rows, cols = [], []
    for user_id, items in partial_train.items():
        for item_id in items:
            rows.append(user_id)
            cols.append(item_id)
    return sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n_users, n_items))


def get_cache_prefix_and_suffix(split_seed=None, split_ratio=None):
    if split_seed is not None and split_ratio is not None:
        return "partial_", f"_seed_{split_seed}_ratio_{int(split_ratio*100)}"
    return "full_", ""


def format_beta_string(beta_val):
    return str(beta_val).replace('.', 'p')



def load_dataset(config):
    from dataloader import Dataset
    return Dataset(path=f"../data/{config['dataset']}")


def create_optimizer(config, param_groups):
    if config['opt'] == 'adam':
        return torch.optim.Adam(param_groups, lr=config['lr'], weight_decay=config['decay'])
    elif config['opt'] == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=config['lr'], weight_decay=config['decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['opt']}")


def save_run_results(config, initial_params, epoch_snapshots, best_model_state,
                     best_epoch, best_ndcg, final_ndcg, final_recall,
                     baseline_ndcg, baseline_recall):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    params_dir = os.path.join(project_root, 'results', 'filter_params')
    os.makedirs(params_dir, exist_ok=True)
    params_filename = f"{config['dataset']}_apsf_{config.get('poly', 'bernstein')}_{config.get('f_init', 'uniform')}_{timestamp}.pkl"

    best_save = dict(best_model_state, epoch=best_epoch, ndcg=best_ndcg,
                     test_ndcg=final_ndcg, test_recall=final_recall)
    with open(os.path.join(params_dir, params_filename), 'wb') as f:
        pickle.dump({'initial': initial_params, 'epochs': epoch_snapshots, 'best': best_save}, f)

    runs_dir = os.path.join(project_root, 'runs', 'train', config['dataset'])
    os.makedirs(runs_dir, exist_ok=True)
    run_summary = {
        'timestamp': timestamp,
        'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                   for k, v in config.items()},
        'baseline_ndcg': baseline_ndcg, 'baseline_recall': baseline_recall,
        'best_epoch': best_epoch, 'val_ndcg': best_ndcg,
        'test_ndcg': final_ndcg, 'test_recall': final_recall,
        'ndcg_improvement_pct': (final_ndcg / baseline_ndcg - 1) * 100,
        'recall_improvement_pct': (final_recall / baseline_recall - 1) * 100,
        'params_file': params_filename,
    }
    with open(os.path.join(runs_dir, f"{timestamp}.json"), 'w') as f:
        json.dump(run_summary, f, indent=2)

    return params_filename
