# Stdlib packages
from abc import ABC, abstractmethod
import json
import os

# Common Py packages
import pandas as pd

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset

################################


class ModelDataset(ABC):
    dfdataset: DFDataset

    def process_config(self, config: dict):
        if type(config) is str: 
            if config.endswith('.json'): 
                eos_filepath = eos.load_file_eos(config)
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            elif config.split('/')[-1].find('.') < 0:
                print(f"WARNING: Config directory supplied rather than file, attempting to load with default filename... ")
                eos_filepath = eos.load_file_eos(os.path.join(config, self.config_filename))
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            else:
                raise IOError(f"ERROR: Config file does not appear to be a json, only JSON is supported currently. ")
            
        for key, value in config.items():
            if hasattr(self, key) and key != "dfdataset": setattr(self, key, value)

    @abstractmethod
    def get_data(self, df: pd.DataFrame, event_weight: str) -> object:
        pass

    @abstractmethod
    def get_train(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        pass

    @abstractmethod
    def get_val(self, fold: int, syst_name: str='nominal', for_eval: bool=False):
        pass

    @abstractmethod
    def get_test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train'):
        pass
    