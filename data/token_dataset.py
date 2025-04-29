import os
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    """
    A Dataset for already-processed inputs saved as .pt files.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(f for f in os.listdir(data_dir) if f.endswith('.pt'))
        self.n_classes = 4
        self.pos_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        self._total_samples = len(self.files)
        
        for f in self.files:
            path = os.path.join(data_dir, f)
            data = torch.load(path, weights_only=False)
            label = f.split("_")[-4:]
            label = [int(x) for x in label]
            label_tensor = torch.tensor(label)
            self.pos_counts += label.sum(dim=0).float()
        neg_counts = self._total_samples - self.pos_counts
        raw_weight = neg_counts / (self.pos_counts + 1e-6)
        self.raw_weight = raw_weight
        self.pos_weight = torch.clamp(raw_weight, min=1.0, max=10.0)
        self.prior_probability = self.pos_counts / self._total_samples
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        name, _ = os.path.splitext(self.files[idx])
        label = name.split("_")[-4:]
        label = [int(x) for x in label]
        label_tensor = torch.tensor(label)
        data['labels'] = label_tensor
        return data