import torch
from torch.utils.data import Dataset
import numpy as np

class MLP_Dataset(Dataset):
    def __init__(self, features, targets, weights):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        # Return the total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample (features, label, weight) at the given index
        return self.X[idx], self.y[idx], self.weights[idx]
