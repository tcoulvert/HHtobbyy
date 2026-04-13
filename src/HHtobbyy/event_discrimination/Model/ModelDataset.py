# Stdlib packages
from abc import ABC, abstractmethod

# Common Py packages
import pandas as pd

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset

################################


class ModelDataset(ABC):
    dfdataset: DFDataset

    def process_config(self, config: dict):
        for key, value in config.items():
            if hasattr(self, key): setattr(self, key, value)

    @abstractmethod
    def get_data(self, df: pd.DataFrame):
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
    