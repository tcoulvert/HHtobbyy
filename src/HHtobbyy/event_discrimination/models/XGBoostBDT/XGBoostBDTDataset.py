# Common Py packages
import numpy as np
import pandas as pd

# ML packages
import xgboost as xgb

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelDataset


class XGBoostBDTDataset(ModelDataset):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        
        # Processes the config
        self.process_config(config)

    #############################################################
    # Common model get
    def get_DMatrix(self, df: pd.DataFrame, event_weight: str):
        return xgb.DMatrix(
            data=df[self.dfdataset.model_vars].values, 
            label=df[f"{self.dfdataset.aux_var_prefix}label1D"].values, 
            weight=np.abs(df[f"{self.dfdataset.aux_var_prefix}{event_weight}"].values),
            missing=self.dfdataset.fill_value, feature_names=self.dfdataset.model_vars
        )
        
    #############################################################
    # Overriding get functions
    def get_data(self, df: pd.DataFrame, event_weight: str):
        return self.get_DMatrix(df, event_weight)
    
    def get_train(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        train_df, _ = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = self.dfdataset.event_weight_var
        else: event_weight = f'{self.dfdataset.event_weight_var}Train'
        return self.get_data(train_df, event_weight)

    def get_val(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        full_train_df = self.dfdataset.get_train(fold, syst_name=syst_name)
        _, val_df = self.dfdataset.train_val_split()(full_train_df, test_size=self.dfdataset.val_split, random_state=self.dfdataset.seed)

        if for_eval: event_weight = self.dfdataset.event_weight_var
        else: event_weight = f'{self.dfdataset.event_weight_var}Train'
        return self.get_data(val_df, event_weight)

    def get_test(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        test_df = self.dfdataset.get_test(fold, syst_name=syst_name, regex=regex)
        
        return self.get_data(test_df, self.dfdataset.event_weight_var)