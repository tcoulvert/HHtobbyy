# Stdlib packages
from abc import ABC, abstractmethod
import gc

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset, batched_writer
from HHtobbyy.event_discrimination.Model import ModelConfig, ModelDataset
from HHtobbyy.event_discrimination.evaluation.evaluation_utils import class_discriminator_columns
from HHtobbyy.workspace_utils import multifold

################################


class Model(ABC):
    dfdataset: DFDataset
    modeldataset: ModelDataset
    modelconfig: ModelConfig

    def train_all_folds(self, **kwargs) -> None:
        multifold(self.train, (), self.dfdataset.n_folds, **kwargs)

    def test_all_folds(self, syst_name: str='nominal', regex: str|list[str]='', **kwargs) -> None:
        multifold(self.test, (syst_name, regex), self.dfdataset.n_folds, **kwargs)

    def predict_all_folds(self, syst_name: str='nominal', regex: str|list[str]='', ckpt_path: str='', **kwargs) -> None:
        multifold(self.predict, (syst_name, regex, ckpt_path), self.dfdataset.n_folds, **kwargs)

    @abstractmethod
    def train(self, fold: int) -> None:
        pass

    @abstractmethod
    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train') -> None:
        pass

    @abstractmethod
    def predict_data(self, data: object, fold: int, ckpt_path: str='') -> np.ndarray:
        pass

    @batched_writer
    def predict(self, fold: int, ckpt_path: str='', **kwargs):
        test_filepaths = self.dfdataset.get_test_filepaths(fold, **kwargs)['test']
        for filepath in test_filepaths:
            self.dfdataset.edit_df(
                filepath, 
                lambda df: pd.DataFrame(
                    self.predict_data(self.modeldataset.get_data(df, self.dfdataset.event_weight_var), fold, ckpt_path=ckpt_path), 
                    columns=[self.dfdataset.aux_var_prefix+col for col in class_discriminator_columns(self.dfdataset.class_sample_map.keys())]
                ),
                **kwargs
            )
