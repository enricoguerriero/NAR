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
        data = torch.load(file_path)
        name, _ = os.path.splitext(self.files[idx])
        label = name.split("_")[-4:]
        label = [int(x) for x in label]
        label_tensor = torch.tensor(label)
        data['labels'] = label_tensor
        return data