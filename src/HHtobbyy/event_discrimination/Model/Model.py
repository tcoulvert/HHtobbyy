# Stdlib packages
from abc import ABC, abstractmethod
from threading import Thread

# Common Py packages
import numpy as np
import pandas as pd

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelConfig, ModelDataset
from HHtobbyy.event_discrimination.evaluation.evaluation_utils import class_discriminator_columns

################################


class Model(ABC):
    dfdataset: DFDataset
    modeldataset: ModelDataset
    modelconfig: ModelConfig

    def train_all_folds(self, parallel: bool=False) -> None:
        for fold in range(self.dfdataset.n_folds):
            if parallel: 
                thread = Thread(target=self.train, name=f"Fold {fold}", args=(fold, ))
                thread.start()
            else:
                self.train(fold)
        thread.join()  # Joins final thread to pause code until all training finished

    def test_all_folds(self, syst_name: str='nominal', regex: str|list[str]='') -> np.ndarray:
        for fold in range(self.dfdataset.n_folds): 
            self.test(fold, syst_name=syst_name, regex=regex)

    def predict_all_folds(self, syst_name: str='nominal', regex: str|list[str]='', **model_kwargs) -> None:
        for fold in range(self.dfdataset.n_folds): 
            self.predict(fold, syst_name=syst_name, regex=regex, model_kwargs=model_kwargs)

    @abstractmethod
    def train(self, fold: int) -> None:
        pass

    @abstractmethod
    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train') -> None:
        pass

    @abstractmethod
    def predict_data(self, data: object, fold: int) -> np.ndarray:
        pass

    def predict(self, fold: int, syst_name: str='nominal', regex: str|list[str]='', model_kwargs: dict={}):
        test_filepaths = self.dfdataset.get_test_filepaths(fold, syst_name=syst_name, regex=regex)['test']
        for filepath in test_filepaths:
            df = self.dfdataset.get_df(filepath)
            data = self.modeldataset.get_data(df, self.dfdataset.event_weight_var, **model_kwargs)
            predictions = self.predict_data(data, fold)
            new_df = pd.DataFrame(predictions, columns=class_discriminator_columns(self.dfdataset.class_sample_map.keys()))
            self.dfdataset.save_df(filepath, new_df)
