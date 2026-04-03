# Common Py packages
import numpy as np
import pandas as pd

# ML packages
import torch
from torch.utils.data import Dataset, DataLoader


from HHtobbyy.event_discrimination.dataset import DFDataset


class MLP_Dataset(DFDataset):
    def __init__(self, dfdataset: DFDataset, config: dict={}):
        super().__init__(dfdataset)

        # Batch sizes for training
        self.train_batch_size = 2048
        self.val_batch_size = 2048
        self.test_batch_size = 4096

        # Number of workers for DataLoader
        self.num_workers = 1

    #############################################################
    # Structure for pytorch dataset
    class MLP_TorchDataset(Dataset):
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
        
    #############################################################
    # Common model get
    def get_MLPTorch(self, df: pd.DataFrame, event_weight: str):
        return self.MLP_TorchDataset(df[self.dfdataset.model_vars].values, df[f"{self.dfdataset.aux_var_prefix}label1D"].values, np.abs(df[f"{self.dfdataset.aux_var_prefix}{event_weight}"].values))
        
    #############################################################
    # Overriding get functions
    def get_train(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        train_df, _ = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = 'eventWeight'
        else: event_weight = 'eventWeightTrain'
        return DataLoader(self.get_MLPTorch(train_df, event_weight), batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=not for_eval)

    def get_val(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        _, val_df = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = 'eventWeight'
        else: event_weight = 'eventWeightTrain'
        return DataLoader(self.get_MLPTorch(val_df, event_weight), batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=not for_eval)

    def get_test(self, fold: int, syst_name: str='nominal', regex: str=''):
        test_df = self.dfdataset.get_test(fold, syst_name=syst_name, regex=regex)
        
        return DataLoader(self.get_MLPTorch(test_df, 'eventWeight'), batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=False)