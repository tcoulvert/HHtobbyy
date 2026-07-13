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
                eos_filepath = eos.load_file_eos(config)
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            elif config.split('/')[-1].find('.') < 0:
                print(f"WARNING: Config directory supplied rather than file, attempting to load with default filename... ")
                eos_filepath = eos.load_file_eos(os.path.join(config, self.config_filename))
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            else:
                raise IOError(f"ERROR: Config file does not appear to be a json, only JSON is supported currently. ")
            
        assert "output_dirpath" in config.keys(), f"ERROR: Required to provide the output_dirpath for the model"
        
        for key, value in config.items():
            if hasattr(self, key) and key != "dfdataset": setattr(self, key, value)
        
        if not eos.file_exists_eos(os.path.join(self.output_dirpath, self.config_filename)): 
            self.model_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
            self.output_dirpath = os.path.join(self.output_dirpath, self.model_time)
            os.makedirs(self.output_dirpath, exist_ok=True)
            self.save_config(self.config_filename)

    def toJSON(self):
        return {**self.__dict__, **{'dfdataset': self.dfdataset.__dict__}}

    def save_config(self, filename: str=config_filename):
        assert filename.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        eos_filepath = eos.save_file_eos(os.path.join(self.output_dirpath, filename))
        with open(eos_filepath, 'w') as f: json.dump(self.toJSON(), f)
        eos.delete_lockfile(eos_filepath)

    @abstractmethod
    def optimize_params(self, model_dataset: ModelDataset, static_params: dict={}, verbose: bool=False):
        pass