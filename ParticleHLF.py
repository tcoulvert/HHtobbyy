# ML packages
import torch
from torch.utils.data import Dataset

class ParticleHLF(Dataset):
    def __init__(self, data_particles, data_hlf, data_y, data_weights):
        self.len = data_y.shape[0]
        self.data_particles = torch.from_numpy(data_particles).float()
        self.data_hlf = torch.from_numpy(data_hlf).float()
        self.data_y = torch.from_numpy(data_y).long()
        self.data_weight = torch.from_numpy(data_weights).float()
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.data_particles[idx], self.data_hlf[idx], self.data_y[idx], self.data_weight[idx])