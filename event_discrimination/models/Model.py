# Stdlib packages
from abc import ABC, abstractmethod

# Workspace packages
from HHtobbyy.event_discrimination.dataset import DFDataset
from HHtobbyy.event_discrimination.dataset import ModelDataset
from HHtobbyy.event_discrimination.models import ModelConfig

################################


class Model(ABC):
    dfdataset: DFDataset
    modeldataset: ModelDataset
    modelconfig: ModelConfig

    @abstractmethod
    def train(self, fold: int):
        pass

    @abstractmethod
    def evaluate(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        pass