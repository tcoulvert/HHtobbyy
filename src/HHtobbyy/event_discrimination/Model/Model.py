# Stdlib packages
from abc import ABC, abstractmethod

# Common Py packages
import numpy as np

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelConfig, ModelDataset

################################


class Model(ABC):
    dfdataset: DFDataset
    modeldataset: ModelDataset
    modelconfig: ModelConfig

    def train_all_folds(self) -> None:
        for fold in range(self.dfdataset.n_folds): 
            self.train(fold)

    def test_all_folds(self, syst_name: str='nominal', regex: str|list[str]='') -> np.ndarray:
        for fold in range(self.dfdataset.n_folds): 
            self.test(fold, syst_name=syst_name, regex=regex)

    def predict_all_folds(self, syst_name: str='nominal', regex: str|list[str]='') -> np.ndarray:
        outputs = []
        for fold in range(self.dfdataset.n_folds): 
            outputs.append(self.predict(fold, syst_name=syst_name, regex=regex))
        return np.concatenate(outputs)

    @abstractmethod
    def train(self, fold: int) -> None:
        pass

    @abstractmethod
    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train') -> list:
        pass

    @abstractmethod
    def predict(self, fold: int, syst_name: str='nominal', regex: str|list[str]='') -> list:
        pass