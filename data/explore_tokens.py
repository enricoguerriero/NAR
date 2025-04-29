import argparse
from collections import Counter
import torch
from token_dataset import TokenDataset

def main(data_dir: str):
    # load dataset
    dataset = TokenDataset(data_dir)
    n_samples = len(dataset)
    print(f"Total samples: {n_samples}\n")

    # peek at one example
    sample = dataset[0]
    print("Data dictionary keys:", list(sample.keys()), "\n")

    # prepare counters
    n_classes = sample['labels'].shape[0]
    pos_counts = torch.zeros(n_classes, dtype=torch.int64)
    shape_counters = {
        'pixel_values': Counter(),
        'attention_mask': Counter(),
        'input_ids': Counter(),
    }

    # iterate
    for item in dataset:
        labels = item['labels']
        pos_counts += labels
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

    # weight computation
    pos_weight, prior_prob = dataset.weight_computation()
    print("\nBCEWithLogitsLoss pos_weight:", pos_weight.tolist())
    print("Class prior probabilities:", prior_prob.tolist(), "\n")

    # tensor shape distributions
    print("Tensor shape distributions:")
    for k, counter in shape_counters.items():
        print(f"\n  {k}:")
        for shape, count in counter.items():
            print(f"    {shape}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore a directory of .pt files with TokenDataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder containing your .pt files",
    )
    args = parser.parse_args()
    main(args.data_dir)
