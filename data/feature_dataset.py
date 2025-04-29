import os
import torch
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    """
    A Dataset for already-processed inputs saved as .pt files.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        all_files = sorted(os.listdir(data_dir))
        self.files = []
        for f in all_files:
            path = os.path.join(data_dir, f)
            data = torch.load(path, weights_only=False)
            features = data['pixel_values_videos']  # or whatever your key is
            if features.shape[0] == 2:
                self.files.append(f)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        return data
    
    def weight_computation(self):
        """
        Computes the pos_weight vector for BCEWithLogitsLoss.
        
        Returns:
            pos_weight (torch.Tensor): Tensor of shape (n_classes,) for BCEWithLogitsLoss.
        """
        n_classes = 4
        pos_counts = torch.zeros(n_classes)
        
        for item in self:
            label = item['labels']
            label_tensor = label.clone().detach().float()
            if label_tensor.dim() == 2:
                label_sum = label_tensor.sum(dim=0)
            else:
                label_sum = label_tensor
            pos_counts += label_sum

        total = len(self)
        neg_counts = total - pos_counts

        # Avoid division by zero
        raw_weight = neg_counts / (pos_counts + 1e-6)

        pos_weight = torch.clamp(raw_weight, max=10.0)

        return pos_weight

    