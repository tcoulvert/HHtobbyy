# Stdlib packages
import datetime
import json
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

    def process_config(self, config: str|dict):
        if type(config) is str: 
            if config.endswith('.json'): 
                with open(eos.load_file_eos(config), 'r') as f: config = json.load(f)
            elif config.split('/')[-1].find('.') < 0:
                print(f"WARNING: Config directory supplied rather than file, attempting to load with default filename... ")
                with open(eos.load_file_eos(os.path.join(config, self.config_filename)), 'r') as f: config = json.load(f)
            else:
                raise IOError(f"ERROR: Config file does not appear to be a json, only JSON is supported currently. ")
            
        assert "output_dirpath" in config.keys(), f"ERROR: Required to provide the output_dirpath for the model"
        
        for key, value in config.items():
            if hasattr(self, key) and key != "dfdataset": setattr(self, key, value)
        
        if not os.path.exists(self.output_dirpath) or self.config_filename not in os.listdir(self.output_dirpath): 
            self.model_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
            self.output_dirpath = os.path.join(self.output_dirpath, self.model_time)
            os.makedirs(self.output_dirpath, exist_ok=True)
            self.save_config(self.config_filename)

    def toJSON(self):
        return {**self.__dict__, **{'dfdataset': self.dfdataset.__dict__}}

    def save_config(self, filename: str=config_filename):
        assert filename.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        with open(eos.save_file_eos(os.path.join(self.output_dirpath, filename)), 'w') as f: json.dump(self.toJSON(), f)

    @abstractmethod
    def optimize_params(self, model_dataset: ModelDataset, static_params: dict={}, verbose: bool=False):
        pass