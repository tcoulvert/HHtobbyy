# Stdlib packages
import datetime
import os
from abc import ABC, abstractmethod

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.dataset import DFDataset
from HHtobbyy.event_discrimination.dataset import ModelDataset

################################


class ModelConfig(ABC):
    dfdataset: DFDataset

    def process_config(self, config: dict):
        assert "output_dirpath" in config.keys(), f"ERROR: Required to provide the output_dirpath for the model"

        if 'model_time' not in config: 
            config['model_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
        self.output_dirpath = os.path.join(config.output_dirpath, config.model_time)

        self.dfdataset_dirpath = self.dfdataset.output_dirpath
        
        for key, value in config.items():
            if hasattr(self, key): setattr(self, key, value)

    def save_config(self, filepath: str):
        assert filepath.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        eos.eos_save_file(self.__dict__, filepath)

    @abstractmethod
    def optimize_params(self, model_dataset: ModelDataset, static_params: dict={}, verbose: bool=False):
        pass