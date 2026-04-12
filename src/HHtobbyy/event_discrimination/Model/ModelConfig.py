# Stdlib packages
import datetime
import os
from abc import ABC, abstractmethod

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelDataset

################################


class ModelConfig(ABC):
    dfdataset: DFDataset
    config_filename = "model_config.json"

    def process_config(self, config: dict):
        assert "output_dirpath" in config.keys(), f"ERROR: Required to provide the output_dirpath for the model"
        
        for key, value in config.items():
            if hasattr(self, key): setattr(self, key, value)

        print()
        
        if self.config_filename not in os.listdir(self.output_dirpath): 
            self.model_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
            self.output_dirpath = os.path.join(self.output_dirpath, self.model_time)
            os.makedirs(self.output_dirpath, exist_ok=True)
            self.save_config(self.config_filename)

    def toJSON(self):
        return {**self.__dict__, **{'dfdataset': self.dfdataset.__dict__}}

    def save_config(self, filename: str=config_filename):
        assert filename.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        eos.save_file_eos(self.toJSON(), os.path.join(self.output_dirpath, filename))

    @abstractmethod
    def optimize_params(self, model_dataset: ModelDataset, static_params: dict={}, verbose: bool=False):
        pass