# Common Py packages
import numpy as np
import pandas as pd

# ML packages
from torch.utils.data import DataLoader

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelDataset
from HHtobbyy.event_discrimination.models.MLP.MLPTorchDataset import MLPTorchDataset


class MLPDataset(ModelDataset):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset

        # Batch sizes for training
        self.train_batch_size = 2048
        self.val_batch_size = 2048
        self.test_batch_size = 4096

        # Number of workers for DataLoader
        self.num_workers = 15

        # Processes the config
        self.process_config(config)
        

    #############################################################
    # Common model get
    def get_MLPTorch(self, df: pd.DataFrame, event_weight: str):
        return MLPTorchDataset(
            df[self.dfdataset.model_vars].to_numpy(), 
            df[f"{self.dfdataset.aux_var_prefix}label1D"].to_numpy(), 
            np.abs(df[f"{self.dfdataset.aux_var_prefix}{event_weight}"].to_numpy())
        )
        

    #############################################################
    # Overriding get functions
    def get_data(self, df: pd.DataFrame, event_weight: str, batch_size: int=2048, shuffle: bool=False):
        return DataLoader(self.get_MLPTorch(df, event_weight), batch_size=batch_size, num_workers=self.num_workers, shuffle=shuffle)
    
    def get_train(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        train_df, _ = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = self.dfdataset.event_weight_var
        else: event_weight = f'{self.dfdataset.event_weight_var}Train'
        return self.get_data(train_df, event_weight=event_weight, batch_size=self.train_batch_size, shuffle=True)

    def get_val(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        _, val_df = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = self.dfdataset.event_weight_var
        else: event_weight = f'{self.dfdataset.event_weight_var}Train'
        return self.get_data(val_df, event_weight=event_weight, batch_size=self.val_batch_size, shuffle=True)

    def get_test(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        test_df = self.dfdataset.get_test(fold, syst_name=syst_name, regex=regex)
        
        return self.get_data(test_df, event_weight=self.dfdataset.event_weight_var, batch_size=self.test_batch_size, shuffle=False)
