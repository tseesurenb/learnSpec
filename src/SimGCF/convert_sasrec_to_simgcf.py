#!/usr/bin/env python3
"""
Convert SASRec ml-1m.txt format to SimGCF train.txt/test.txt format
SASRec format: user_id item_id (one per line, in temporal order)
SimGCF format: user_id item_1 item_2 ... (all items per user on one line)
"""

import sys
from collections import defaultdict

def convert_sasrec_to_simgcf(sasrec_file, output_dir):
    """
    Convert SASRec format to SimGCF format.
    Following SASRec's split: last item → test, rest → train
    """
    # Read all interactions
    user_items = defaultdict(list)
    
    print("Reading SASRec file...")
    with open(sasrec_file, 'r') as f:
        for line in f:
            user, item = map(int, line.strip().split())
            user_items[user].append(item)
    
    print(f"Found {len(user_items)} users")
    
    # Split into train and test
    train_data = {}
    test_data = {}
    
    for user, items in user_items.items():
        if len(items) < 4:  # Following SASRec logic
            # Users with <4 items: all go to train
            train_data[user] = items
            test_data[user] = []
        else:
            # Last item → test, rest → train
            train_data[user] = items[:-1]
            test_data[user] = [items[-1]]
    
    # Write train.txt
    train_file = f"{output_dir}/train.txt"
    print(f"Writing {train_file}...")
    with open(train_file, 'w') as f:
        for user in sorted(train_data.keys()):
            if train_data[user]:  # Only write if user has items
                items_str = ' '.join(map(str, train_data[user]))
                f.write(f"{user} {items_str}\n")
    
    # Write test.txt
    test_file = f"{output_dir}/test.txt"
    print(f"Writing {test_file}...")
    with open(test_file, 'w') as f:
        for user in sorted(test_data.keys()):
            if test_data[user]:  # Only write if user has test items
                items_str = ' '.join(map(str, test_data[user]))
                f.write(f"{user} {items_str}\n")
    
    # Print statistics
    train_interactions = sum(len(items) for items in train_data.values())
    test_interactions = sum(len(items) for items in test_data.values())
    
    print("\nConversion complete!")
    print(f"Train: {train_interactions} interactions")
    print(f"Test: {test_interactions} interactions")
    print(f"Total: {train_interactions + test_interactions} interactions")
    print(f"Users with test data: {sum(1 for items in test_data.values() if items)}")

if __name__ == "__main__":
    # Paths
    sasrec_file = "../../baselines/srcs/SASRec.pytorch/python/data/ml-1m.txt"
    output_dir = "../../data/ml-1m"
    
    convert_sasrec_to_simgcf(sasrec_file, output_dir)