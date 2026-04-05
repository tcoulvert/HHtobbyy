# Stdlib packages
import argparse
import logging

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.dataset import DFDataset
from HHtobbyy.event_discrimination.models import Model, MLP, map_model_to_Model

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Run event discriminator training and evaluation")
parser.add_argument(
    "dfdataset_config", 
    type=str,
    help="Full filepath to DFdataset config file"
)
parser.add_argument(
    "model_config", 
    type=str,
    help="Full filepath to model config file"
)
parser.add_argument(
    "model", 
    choices=['XGBoostBDT', 'MLP'],
    help="Model architecture to train (see models)"
)

################################



def main(dfdataset: DFDataset, model: Model):
    model.train_all_folds()

    predictions = model.evaluate_all_folds()

    

if __name__ == "__main__":
    args = parser.parse_args
    dfdataset = DFDataset(args.dfdataset_config)
    model = map_model_to_Model(args.model)(dfdataset, args.model_config)

    main(dfdataset, model)
