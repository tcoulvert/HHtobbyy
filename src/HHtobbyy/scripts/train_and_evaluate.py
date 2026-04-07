# Stdlib packages
import argparse
import logging

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models import map_model_to_Model
from HHtobbyy.workspace_utils.retrieval_utils import get_input_filepaths

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
parser.add_argument(
    "--eras", 
    type=str|list[str],
    default='',
    help="Era filepaths to train/evaluate"
)
parser.add_argument(
    "--filepaths", 
    type=str|list[str],
    default='',
    help="Filepaths to train/evaluate, overrides eras"
)

################################




def main(dfdataset: DFDataset, model: Model, filepaths: list):
    # Building train DFDataset
    dfdataset.make_all_train(filepaths)

    # Training the model
    model.train_all_folds()

    # Building test DFDataset
    dfdataset.make_all_test(filepaths)

    # Evaluating the model
    predictions = model.evaluate_all_folds()



if __name__ == "__main__":
    args = parser.parse_args
    assert args.eras != '' or args.filepaths != '', f"ERROR: Must provide either era filepath(s) or direct filepath(s)"

    dfdataset = DFDataset(args.dfdataset_config)
    model = map_model_to_Model(args.model)(dfdataset, args.model_config)

    if type(args.filepaths) is str and args.filepaths != '':
        filepaths = eos.load_file_eos(args.filepaths)
    elif type(args.filepaths) is list:
        filepaths = args.filepaths
    else:
        filepaths = get_input_filepaths(args.eras, dfdataset.class_sample_map, regex=f"*{dfdataset.filepostfix}")

    main(dfdataset, model, filepaths)
