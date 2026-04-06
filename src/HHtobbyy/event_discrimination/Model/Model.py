# Stdlib packages
from abc import ABC, abstractmethod

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

    def evaluate_all_folds(self, syst_name: str='nominal', regex: str|list[str]='') -> list:
        outputs = []
        for fold in range(self.dfdataset.n_folds): 
            outputs.append(self.evaluate(fold, syst_name=syst_name, regex=regex))
        return outputs

    @abstractmethod
    def train(self, fold: int) -> None:
        pass

    @abstractmethod
    def evaluate(self, fold: int, syst_name: str='nominal', regex: str|list[str]='') -> list:
        pass