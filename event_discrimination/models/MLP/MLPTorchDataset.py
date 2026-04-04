# Common Py packages
import numpy as np

# ML packages
import torch
from torch.utils.data import Dataset


#############################################################
# Structure for pytorch dataset
class MLPTorchDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        # Return the total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample (features, label, weight) at the given index
        return self.X[idx], self.y[idx], self.weights[idx]
        