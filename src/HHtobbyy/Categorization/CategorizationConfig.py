# Stdlib packages
import datetime
import os

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.evaluation import transform_preds_options, transform_preds_func
from .categorization_utils import *

class CategorizationConfig:
    def __init__(self, dfdataset: DFDataset, config: dict):
        # DFDataset dirpath -- store the categorization nearby
        self.output_dirpath = os.path.join(dfdataset.output_dirpath, 'categories')

        # Time of running categorization
        self.cat_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'

        # Base filename for output categories
        self.cat_filename = f"{self.cat_time}_categories.json"
        
        # Performs category optimization using fit to sidebands for estimation of non-resonant background in SR
        self.opt_sideband = "mc"  # 'none', 'data', 'mc'

        # Defines the discriminator to use for categorization, discriminators are implemented in evaluation_utils
        self.discriminator = transform_preds_options()[0]

        # Figure of Merit function to maximize
        self.maximization_func = "s_over_b"

        # Method to search for best cuts
        self.cat_method = "grid_search"

        # Options for chosen method
        self.method_options = {'n_steps': 10, 'n_zoom': 6}

        # Number of categories
        self.n_cats = 3

        # Samples for signal
        self.signal_samples = ['GluGlu*HH*kl-1p00']

        # Res samples regex to evaluate
        self.res_samples = ['ttH', 'bbH', 'VH', 'W*HToGG', 'ZHToGG', 'GluGluHToGG', 'VBFHToGG']

        # nonRes samples regex to evaluate
        self.nonres_samples = ['TTGG', 'GJet', 'GGJets', 'DDQCDGJet', 'SherpaNLO']

        # Diphoton mass values for Signal Region (SR)
        self.SR_masscut = [122.5, 127.]

        # Diphoton mass values for Sideband (SB)
        self.SB_masscut = [100., 180.]

        # Blinds data in any fits or plots
        self.blind_data = True

        # Diphoton mass fit parameters (start, stop, step)
        self.fit_bins = self.SB_masscut + [5.]

        # Processes the input config file
        self.process_config(config)


    def get_fom(self):
        if self.maximization_func == "s_over_b": fom_s_over_b
        elif self.maximization_func == "s_over_sqrt_b": fom_s_over_sqrt_b
        else: raise NotImplementedError(f"Maximization method not yet implemented, use \'s_over_b\' or \'s_over_sqrt_b\'.")

    def get_catmethod(self):
        if self.cat_method == "grid_search": grid_search
        else: raise NotImplementedError(f"Maximization method not yet implemented, use \'grid_search\'.")

    def get_transform(self):
        self.transform_names, _, self.cutdir = transform_preds_func(self.dfdataset.class_sample_map, self.discriminator)

    def get_optcolumns(self):
        self.opt_columns_map = {self.dfdataset.aux_var_prefix + col: col for col in self.transform_names+['mass', 'eventWeight']}

    def get_transform_dict(self):
        return {'names': self.transform_names, 'cutdir': self.cutdir}

    def process_config(self, config: dict):
        for key, value in config.items():
            if hasattr(self, key): setattr(self, key, value)
        
        self.get_transform()
        self.get_optcolumns()
        self.save_config()

    def toJSON(self):
        return {key: value for key, value in self.__dict__.items()}

    def save_config(self):
        assert self.cat_filename.endswith('.json'), f"ERROR: Currently only supporting \'json\' type config serializations"
        
        os.makedirs(self.output_dirpath, exist_ok=True)
        eos.save_file_eos(self.toJSON(), os.path.join(self.output_dirpath, self.cat_filename))
            