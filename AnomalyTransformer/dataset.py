import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size=100):
        self.windows = np.array([
            data[i:i + window_size]
            for i in range(len(data) - window_size + 1)
        ])

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)
    
def get_dataloader(data, window_size=100, batch_size=32, shuffle=True):
    dataset = SlidingWindowDataset(data, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    