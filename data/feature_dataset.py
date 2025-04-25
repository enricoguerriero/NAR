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
        data = torch.load(file_path)
        label = self.files[idx].split("_")[-4:]
        label = [int(x) for x in label]
        return data, label
    