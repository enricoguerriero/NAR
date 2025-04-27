import os
import torch
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    """
    A Dataset for already-processed inputs saved as .pt files.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        label = self.files[idx].split("_")[-4:]
        label = [int(x) for x in label]
        return data, label
    
    def weight_computation(self):
        """
        Computes the pos_weight vector for BCEWithLogitsLoss.
        
        Returns:
            pos_weight (torch.Tensor): Tensor of shape (n_classes,) for BCEWithLogitsLoss.
        """
        n_classes = 4
        pos_counts = torch.zeros(n_classes)
        
        for _, label in self:
            label_tensor = torch.tensor(label, dtype=torch.float32)
            pos_counts += label_tensor

        total = len(self)
        neg_counts = total - pos_counts

        # Avoid division by zero
        raw_weight = neg_counts / (pos_counts + 1e-6)

        pos_weight = torch.clamp(raw_weight, max=10.0)

        return pos_weight

    