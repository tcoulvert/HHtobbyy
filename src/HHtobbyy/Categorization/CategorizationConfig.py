# Stdlib packages
import datetime
import os

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.evaluation import transform_preds_options
from .categorization_utils import *

class CategorizationConfig:
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.output_dirpath = os.path.join(dfdataset.output_dirpath, 'categories')

        self.cat_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'

        self.cat_filename = f"{self.cat_time}_categories.json"
        
        self.opt_sideband = "sherpa"  # 'none', 'data', 'mc', 'sherpa'

        self.discriminator = transform_preds_options()[0]

        self.signal_index = 0

        self.maximization_func = "s_over_b"

        self.cat_method = "grid_search"

        self.process_config(config)

    def get_fom(self):
        if self.maximization_func == "s_over_b": fom_s_over_b
        elif self.maximization_func == "s_over_sqrt_b": fom_s_over_sqrt_b
        else: raise NotImplementedError(f"Maximization method not yet implemented, use \'s_over_b\' or \'s_over_sqrt_b\'.")

    def get_catmethod(self):
        if self.cat_method == "grid_search": grid_search
        else: raise NotImplementedError(f"Maximization method not yet implemented, use \'grid_search\'.")

    def process_config(self, config: dict):
        for key, value in config.items():
            if hasattr(self, key): setattr(self, key, value)
            
        os.makedirs(self.output_dirpath, exist_ok=True)

    def save_config(self):
        assert self.cat_filename.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        eos.save_file_eos(self.__dict__, os.path.join(self.output_dirpath, self.cat_filename))
            