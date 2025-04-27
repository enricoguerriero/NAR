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
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, weights_only=True)
        name, _ = os.path.splitext(self.files[idx])
        label = name.split("_")[-4:]
        label = [int(x) for x in label]
        label_tensor = torch.tensor(label)
        data['labels'] = label_tensor
        return data
    
    def weight_computation(self):
        """
        Computes the pos_weight vector for BCEWithLogitsLoss.
        
        Returns:
            pos_weight (torch.Tensor): Tensor of shape (n_classes,) for BCEWithLogitsLoss.
        """
        n_classes = 4
        pos_counts = torch.zeros(n_classes)
        
        for clip in self:
            label_tensor = clip['labels']
            pos_counts += label_tensor

        total = len(self)
        neg_counts = total - pos_counts

        # Avoid division by zero
        raw_weight = neg_counts / (pos_counts + 1e-6)

        pos_weight = torch.clamp(raw_weight, max=10.0)
        prior_probability = pos_counts / total 

        return pos_weight, prior_probability