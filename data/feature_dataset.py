import os
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = []  # List of (file, index_within_file)
        self.n_classes = 4
        
        pos_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        
        all_files = sorted(os.listdir(data_dir))
        for f in all_files:
            path = os.path.join(data_dir, f)
            data = torch.load(path, weights_only=False)
            labels = data['labels']
            pos_counts = torch.zeros(self.n_classes, dtype=torch.float32)
            if labels.shape[0] == 2:
                # Save two entries per file
                self.samples.append((f, 0))
                self.samples.append((f, 1))
                pos_counts += labels.sum(dim=0).float()
                
                
        self._pos_counts = pos_counts
        self._total_samples = len(self)
        
        neg_counts = self._total_samples - pos_counts
        raw_weight = neg_counts / (pos_counts + 1e-6)
        
        self.pos_weight = torch.clamp(raw_weight, min=1.0, max=10.0)
    
                
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

