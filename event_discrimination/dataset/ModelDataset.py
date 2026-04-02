# Stdlib packages
from abc import ABC, abstractmethod

# Workspace packages
from HHtobbyy.event_discrimination.dataset.DFDataset import DFDataset

################################


class ModelDataset(ABC):
    def __init__(dfdataset: DFDataset):
        self.dfdataset = dfdataset

    @abstractmethod
    def get_train(self, fold: int):
        pass

    @abstractmethod
    def get_val(self, fold: int):
        pass

    @abstractmethod
    def get_test(self, fold: int):
        pass