import torch
import numpy as np
import utils


def MSE_train_spectral(validation_data, model, optimizer, batch_size=1000):
    model.train()
    users_with_validation = [u for u, items in validation_data.items() if len(items) > 0]
    if len(users_with_validation) == 0:
        return 0.0

    total_loss = 0.0
    optimizer.zero_grad()

    for batch_start in range(0, len(users_with_validation), batch_size):
        batch_end = min(batch_start + batch_size, len(users_with_validation))
        batch_users = users_with_validation[batch_start:batch_end]

        batch_val_items = set()
        user_val_mappings = {}
        for i, user_id in enumerate(batch_users):
            val_items = validation_data[user_id]
            if len(val_items) > 0:
                user_val_mappings[i] = val_items
                batch_val_items.update(val_items)

        if not batch_val_items:
            continue

        batch_val_items = sorted(list(batch_val_items))
        val_item_to_idx = {item: idx for idx, item in enumerate(batch_val_items)}

        users = torch.as_tensor(batch_users, dtype=torch.long, device=model.device)
        predicted_ratings = model.forward_selective(users, batch_val_items)

        mse_loss = 0
        total_val_interactions = 0
        for i, user_id in enumerate(batch_users):
            if i in user_val_mappings:
                val_items = user_val_mappings[i]
                val_positions = [val_item_to_idx[item] for item in val_items]
                user_val_predictions = predicted_ratings[i, val_positions]
                mse_loss += torch.mean((user_val_predictions - 1.0) ** 2)
                total_val_interactions += len(val_items)

        if total_val_interactions > 0:
            mse_loss = mse_loss / len([u for u in batch_users if len(validation_data[u]) > 0])
        else:
            mse_loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        batch_weight = len(batch_users) / len(users_with_validation)
        scaled_loss = mse_loss * batch_weight
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
