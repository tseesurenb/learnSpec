import torch
import numpy as np
import utils



def _get_neg_arrays(validation_data, users_with_validation, n_items, cache_dir=None, split_seed=42, split_ratio=0.7):
    """Build or load cached negative item arrays per user."""
    import os, pickle

    if cache_dir:
        neg_cache_file = os.path.join(cache_dir, f'neg_arrays_seed{split_seed}_ratio{int(split_ratio*100)}_n{n_items}.pkl')
        if os.path.exists(neg_cache_file):
            with open(neg_cache_file, 'rb') as f:
                return pickle.load(f)

    neg_arrays = {}
    all_items_set = set(range(n_items))
    for u in users_with_validation:
        pos_set = set(validation_data[u])
        neg_arrays[u] = np.array(sorted(all_items_set - pos_set), dtype=np.int32)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(neg_cache_file, 'wb') as f:
            pickle.dump(neg_arrays, f)
        print(f"    Cached negative arrays: {os.path.basename(neg_cache_file)}")

    return neg_arrays


def BPR_train_spectral(validation_data, model, optimizer, batch_size=1000, n_items=None, n_neg=1,
                       cache_dir=None, split_seed=42, split_ratio=0.7):
    model.train()
    users_with_validation = [u for u, items in validation_data.items() if len(items) > 0]
    if len(users_with_validation) == 0:
        return 0.0

    if n_items is None:
        n_items = model.n_items

    user_neg_arrays = _get_neg_arrays(validation_data, users_with_validation, n_items,
                                      cache_dir=cache_dir, split_seed=split_seed, split_ratio=split_ratio)

    total_loss = 0.0
    optimizer.zero_grad()

    for batch_start in range(0, len(users_with_validation), batch_size):
        batch_end = min(batch_start + batch_size, len(users_with_validation))
        batch_users = users_with_validation[batch_start:batch_end]

        # Sample one positive and n_neg negatives per user
        b_users, b_pos, b_neg_list = [], [], []
        for user_id in batch_users:
            val_items = validation_data[user_id]
            if len(val_items) == 0:
                continue
            pos_item = val_items[np.random.randint(len(val_items))]
            neg_items = np.random.choice(user_neg_arrays[user_id], size=n_neg, replace=True)
            b_users.append(user_id)
            b_pos.append(pos_item)
            b_neg_list.append(neg_items.tolist() if n_neg > 1 else [int(neg_items[0])])

        if not b_users:
            continue

        # Collect all unique target items
        all_items = set(b_pos)
        for negs in b_neg_list:
            all_items.update(negs)
        target_items = sorted(all_items)
        item_to_idx = {item: idx for idx, item in enumerate(target_items)}

        users = torch.as_tensor(b_users, dtype=torch.long, device=model.device)
        predicted = model.forward_selective(users, target_items)
        bs = len(b_users)

        pos_idx = [item_to_idx[p] for p in b_pos]
        pos_scores = predicted[range(bs), pos_idx]

        if n_neg == 1:
            neg_idx = [item_to_idx[b_neg_list[i][0]] for i in range(bs)]
            neg_scores = predicted[range(bs), neg_idx]
            bpr_loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()
        else:
            neg_idx = [[item_to_idx[n] for n in negs] for negs in b_neg_list]
            neg_idx_t = torch.tensor(neg_idx, dtype=torch.long, device=model.device)
            neg_scores = torch.gather(predicted, 1, neg_idx_t)
            bpr_loss = torch.nn.functional.softplus(neg_scores - pos_scores.unsqueeze(1)).mean()

        batch_weight = len(b_users) / len(users_with_validation)
        scaled_loss = bpr_loss * batch_weight
        total_loss += scaled_loss.item()
        scaled_loss.backward()

    optimizer.step()
    return total_loss


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
