import os
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = []  # List of (file, index_within_file)
        
        all_files = sorted(os.listdir(data_dir))
        for f in all_files:
            path = os.path.join(data_dir, f)
            data = torch.load(path, weights_only=False)
            features = data['labels']
            if features.shape[0] == 2:
                # Save two entries per file
                self.samples.append((f, 0))
                self.samples.append((f, 1))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, internal_idx = self.samples[idx]
        path = os.path.join(self.data_dir, f)
        data = torch.load(path, weights_only=False)
        sample = {
            'features': data['features'][internal_idx],
            'labels': data['labels'][internal_idx]
        }
        return sample

    def weight_computation(self):
        n_classes = 4
        pos_counts = torch.zeros(n_classes)
        
        for item in self:
            label = item['labels'].float()
            pos_counts += label

        total = len(self)
        neg_counts = total - pos_counts

        raw_weight = neg_counts / (pos_counts + 1e-6)
        pos_weight = torch.clamp(raw_weight, max=10.0)
        return pos_weight
