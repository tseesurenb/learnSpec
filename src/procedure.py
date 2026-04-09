import torch
import numpy as np
import utils



def train_spectral(validation_data, model, optimizer, batch_size=1000, loss_type='bpr'):
    model.train()
    users_with_validation = [u for u, items in validation_data.items() if len(items) > 0]
    if len(users_with_validation) == 0:
        return 0.0

    n_items = model.n_items
    user_pos_sets = {u: set(validation_data[u]) for u in users_with_validation}

    total_loss = 0.0
    optimizer.zero_grad()

    for batch_start in range(0, len(users_with_validation), batch_size):
        batch_end = min(batch_start + batch_size, len(users_with_validation))
        batch_users = users_with_validation[batch_start:batch_end]

        b_users, b_pos, b_neg = [], [], []
        for user_id in batch_users:
            val_items = validation_data[user_id]
            if len(val_items) == 0:
                continue
            pos_item = val_items[np.random.randint(len(val_items))]
            neg = np.random.randint(n_items)
            while neg in user_pos_sets[user_id]:
                neg = np.random.randint(n_items)
            b_users.append(user_id)
            b_pos.append(pos_item)
            b_neg.append(neg)

        if not b_users:
            continue

        target_items = sorted(set(b_pos + b_neg))
        item_to_idx = {item: idx for idx, item in enumerate(target_items)}

        users = torch.as_tensor(b_users, dtype=torch.long, device=model.device)
        predicted = model.forward_selective(users, target_items)
        bs = len(b_users)

        pos_idx = [item_to_idx[p] for p in b_pos]
        neg_idx = [item_to_idx[n] for n in b_neg]
        pos_scores = predicted[range(bs), pos_idx]
        neg_scores = predicted[range(bs), neg_idx]

        if loss_type == 'bce':
            loss = (-torch.nn.functional.logsigmoid(pos_scores)
                    - torch.nn.functional.logsigmoid(-neg_scores)).mean()
        else:  # bpr
            loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()

        batch_weight = bs / len(users_with_validation)
        scaled_loss = loss * batch_weight
        total_loss += scaled_loss.item()
        scaled_loss.backward()

    optimizer.step()
    return total_loss


# Legacy aliases
BPR_train_spectral = lambda *a, **kw: train_spectral(*a, **kw, loss_type='bpr')
BCE_train_spectral = lambda *a, **kw: train_spectral(*a, **kw, loss_type='bce')


def evaluate(dataset, model, split='test', batch_size=1000):
    eval_dict = dataset.valDict if split == 'val' else dataset.testDict
    max_K = 20
    results = {'precision': np.zeros(1), 'recall': np.zeros(1), 'ndcg': np.zeros(1)}

    users = list(eval_dict.keys())
    if len(users) == 0:
        return results

    rating_list = []
    groundTrue_list = []

    for i in range(0, len(users), batch_size):
        batch_users = users[i:i + batch_size]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [eval_dict[u] for u in batch_users]

        rating = model.getUsersRating(batch_users).to(model.device)

        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend([int(x) for x in items])
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = torch.topk(rating, k=max_K)

        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)

    for ratings, truths in zip(rating_list, groundTrue_list):
        r = utils.getLabel(truths, ratings.tolist())
        ret = utils.RecallPrecision_ATk(truths, r, max_K)
        results['recall'] += ret['recall']
        results['precision'] += ret['precision']
        results['ndcg'] += utils.NDCGatK_r(truths, r, max_K)

    n = float(len(users))
    results['recall'] /= n
    results['precision'] /= n
    results['ndcg'] /= n
    return results


def evaluate_baseline(dataset, config):
    from model import LearnSpecCF
    model = LearnSpecCF(dataset.UserItemNet, config, use_cache=True).to(config['device'])
    model.eval()
    with torch.no_grad():
        results = evaluate(dataset, model, split='test', batch_size=config['batch_size'])
    del model
    return results['ndcg'][0], results['recall'][0]


# Legacy aliases
Test = lambda dataset, model, epoch, config=None, batch_size=1000: evaluate(dataset, model, split='test', batch_size=batch_size)
Test_val = lambda dataset, model, epoch, config=None, batch_size=1000: evaluate(dataset, model, split='val', batch_size=batch_size)
