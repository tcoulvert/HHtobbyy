# Stdlib packages
import argparse
import json
import logging

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models import map_model_to_Model
from HHtobbyy.workspace_utils.retrieval_utils import get_input_filepaths
from HHtobbyy.Categorization import Categorization

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Run event discriminator training and evaluation")
parser.add_argument(
    "dfdataset_config", 
    type=str,
    help="Full filepath to DFdataset config file(s), if passing multiple delimeter is \', \'"
)

################################




def main(dfdataset: DFDataset):
    # Categorizing the model
    # cat = Categorization(dfdataset, {"discriminator": "Boost1D", "method_options": {'step_size': 0.005, 'patience': 1}, "min_nonres_sideband": 4., "n_cats": 1, "maximization_func": "fom_zscore"})
    # cat = Categorization(dfdataset, {"discriminator": "SnT2D", "method_options": {'step_size': 0.0005, 'patience': 1}, "min_nonres_sideband": 10., "n_cats": 3, "maximization_func": "fom_zscore"})
    cat = Categorization(dfdataset, {"discriminator": "DttH-DQCD", "method_options": {'step_size': 0.0005, 'patience': 1}, "min_nonres_sideband": 10., "n_cats": 3, "maximization_func": "fom_zscore"})
    cat.run()


if __name__ == "__main__":
    args = parser.parse_args()
    
    dfdataset = DFDataset(args.dfdataset_config)

    main(dfdataset)
