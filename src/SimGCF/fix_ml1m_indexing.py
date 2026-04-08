#!/usr/bin/env python3
"""
Fix ml-1m indexing to be 0-based for SimGCF
SASRec uses 1-based indexing, but SimGCF expects 0-based
"""

def convert_to_zero_based(input_file, output_file):
    """Convert 1-based indexing to 0-based indexing"""
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = list(map(int, line.strip().split()))
            # Convert user ID and all item IDs to 0-based
            user = parts[0] - 1
            items = [item - 1 for item in parts[1:]]
            
            # Write converted line
            fout.write(f"{user} {' '.join(map(str, items))}\n")
    
    print(f"Converted {input_file} -> {output_file}")

if __name__ == "__main__":
    # Convert train and test files
    convert_to_zero_based("../../data/ml-1m/train.txt", "../../data/ml-1m/train_0based.txt")
    convert_to_zero_based("../../data/ml-1m/test.txt", "../../data/ml-1m/test_0based.txt")
    
    # Rename files
    import os
    os.rename("../../data/ml-1m/train.txt", "../../data/ml-1m/train_1based.txt")
    os.rename("../../data/ml-1m/test.txt", "../../data/ml-1m/test_1based.txt")
    os.rename("../../data/ml-1m/train_0based.txt", "../../data/ml-1m/train.txt")
    os.rename("../../data/ml-1m/test_0based.txt", "../../data/ml-1m/test.txt")
    
    print("Files renamed: original files saved as *_1based.txt")