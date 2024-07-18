import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        # Generate example data
        self.observed_data = np.random.rand(1000, 50, 128)  # 1000 samples, 50 time points, 128 features
        self.observed_mask = np.random.randint(0, 2, (1000, 50, 128))  # Random mask
        self.observed_tp = np.random.rand(1000, 50)  # Random time points

        # Convert to PyTorch tensors
        self.observed_data = torch.tensor(self.observed_data, dtype=torch.float32)
        self.observed_mask = torch.tensor(self.observed_mask, dtype=torch.float32)
        self.observed_tp = torch.tensor(self.observed_tp, dtype=torch.float32)

    def __len__(self):
        return len(self.observed_data)

    def __getitem__(self, idx):
        return self.observed_data[idx], self.observed_mask[idx], self.observed_tp[idx]
