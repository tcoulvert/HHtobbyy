# Stdlib packages
import argparse

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models import map_model_to_Model

################################


parser = argparse.ArgumentParser(description="Run event discriminator training and evaluation")
parser.add_argument(
    "dfdataset_config", 
    type=str,
    help="Full filepath to DFdataset config file(s), if passing multiple delimeter is \', \'"
)
parser.add_argument(
    "model_config", 
    type=str,
    help="Full filepath to model config file(s), if passing multiple delimeter is \', \'"
)
parser.add_argument(
    "model", 
    type=str,
    choices=['MLP', 'XGBoostBDT'],
    help="Types of models currently implemented"
)

################################


if __name__ == "__main__":
    args = parser.parse_args()

    dfdataset = DFDataset(args.dfdataset_config)
    print(type(dfdataset))

    model = map_model_to_Model(args.model)(dfdataset, args.model_config)
    print(type(model))
