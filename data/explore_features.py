#!/usr/bin/env python
import os
import argparse
from collections import Counter
import torch
from feature_dataset import FeatureDataset  

def main(data_dir: str):
    # load dataset
    dataset = FeatureDataset(data_dir)
    n_samples = len(dataset)
    print(f"Total samples: {n_samples}\n")
    
    check_label_shapes(dataset)

    # peek at one example to get keys
    sample = dataset[0]
    keys = list(sample.keys())
    print("Data dictionary keys:", keys, "\n")

    # prepare counters
    n_classes = sample['labels'].shape[0]
    pos_counts = torch.zeros(n_classes, dtype=torch.int64)
    # build a counter for each tensor key except labels
    shape_counters = {k: Counter() for k in keys if k != 'labels'}

    # iterate through dataset
    for item in dataset:
        labels = item['labels'].squeeze(0)
        print(f"Labels shape: {labels.shape}", flush=True)
        print(f"Labels: {labels.tolist()}", flush=True)
        pos_counts += labels.to(torch.int64)
        for k in shape_counters:
            tensor = item[k]
            shape_counters[k][tuple(tensor.shape)] += 1

    # negative counts and ratios
    neg_counts = n_samples - pos_counts
    print("Label summary per class:")
    for i in range(n_classes):
        pos = pos_counts[i].item()
        neg = neg_counts[i].item()
        ratio = pos / n_samples
        print(f"  Class {i:>1} → pos: {pos:>4}, neg: {neg:>4}, pos_ratio: {ratio:.3f}")

    # compute BCEWithLogitsLoss weights and priors
    pos_weight, prior_prob = dataset.weight_computation()

    print("\nBCEWithLogitsLoss pos_weight:", pos_weight.tolist())
    print("Class prior probabilities:", prior_prob.tolist(), "\n")

    # tensor shape distributions
    print("Tensor shape distributions:")
    for k, counter in shape_counters.items():
        print(f"\n  {k}:")
        for shape, count in counter.items():
            print(f"    {shape}: {count} samples")

def check_label_shapes(dataset):
    print("Checking label shapes across dataset...")
    shape_counter = Counter()
    for i, item in enumerate(dataset):
        shape = tuple(item['labels'].shape)
        shape_counter[shape] += 1
    print("Label shapes found:")
    for shape, count in shape_counter.items():
        print(f"  Shape {shape}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore a directory of .pt files with FeatureDataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder containing your .pt files",
    )
    args = parser.parse_args()
    main(args.data_dir)
