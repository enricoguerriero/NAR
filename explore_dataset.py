#!/usr/bin/env python3
"""
A fast exploration script for the FeatureDataset class.
Usage:
    python explore_dataset.py --data_dir /path/to/pt_files --batch_size 128 --num_workers 8
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import your dataset class (adjust the import path as needed)
from data.feature_dataset import FeatureDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Explore the FeatureDataset quickly.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing .pt files for FeatureDataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--preview_samples", type=int, default=5,
        help="Number of individual samples to preview"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize dataset
    dataset = FeatureDataset(args.data_dir)
    n_samples = len(dataset)
    print(f"Total samples in dataset: {n_samples}")

    # Compute positive class weights
    print("Computing pos_weight for BCEWithLogitsLoss...")
    pos_weight = dataset.pos_weight
    print(f"pos_weight tensor: {pos_weight}\n")

    # Quick preview of raw sample structures
    print(f"Previewing first {args.preview_samples} samples:")
    for idx in range(min(args.preview_samples, n_samples)):
        item = dataset[idx]
        print(f"Sample {idx}: keys = {list(item.keys())}")
        for key, tensor in item.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  - {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                      f"min={tensor.min().item()}, max={tensor.max().item()}")
        print()

    # Use DataLoader for batch-level inspection
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    print("Iterating data loader to inspect batch shapes:")
    dimensions = {}
    for batch in loader:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                if key not in dimensions:
                    dimensions[key] = []
                dimensions[key].append(tensor.shape)
    # check if all the shapes for a specific key are the same
    for key, shapes in dimensions.items():
        if len(set(shapes)) == 1:
            print(f"  - {key}: consistent shape {shapes[0]}")
        else:
            print(f"  - {key}: inconsistent shapes {set(shapes)}")

    print("\nScript completed successfully.")

if __name__ == "__main__":
    main()
