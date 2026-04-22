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
from HHtobbyy.Categorization import Categorization

################################


logger = logging.getLogger(__name__)
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
    "--eras", 
    type=str,
    default='',
    help="Era filepaths to train/evaluate"
)
parser.add_argument(
    "--filepaths", 
    type=str,
    default='',
    help="Filepaths to train/evaluate, overrides eras"
)
parser.add_argument(
    "--submission", 
    type=str,
    default='iterative',
    choices=['iterative', 'parallel'],
    help="How to run script"
)

################################




def main(dfdataset: DFDataset, model: Model, filepaths: list, **kwargs):
    # Building train DFDataset
    dfdataset.make_all_train(filepaths, **kwargs)

    # Training the model
    model.train_all_folds(**kwargs)

    # Building test DFDataset
    dfdataset.make_all_test(filepaths, **kwargs)

    # Evaluating the model
    model.predict_all_folds(**kwargs)

    # Categorizing the model
    cat = Categorization(dfdataset, {"discriminator": "3D"})
    cat.run()


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.eras != '' or args.filepaths != '', f"ERROR: Must provide either era filepath(s) or direct filepath(s)"

    dfdataset = DFDataset(args.dfdataset_config)

    model_config = eos.load_file_eos(dict, args.model_config)
    model = map_model_to_Model(model_config['model'])(dfdataset, model_config)

    if args.filepaths != '' and len(args.filepaths.split(', ')) == 1:
        filepaths = eos.load_file_eos(args.filepaths)
    elif len(args.filepaths.split(', ')) > 1:
        filepaths = args.filepaths.split(', ')
    else:
        filepaths = get_input_filepaths(args.eras.split(', ') if len(args.eras.split(', ')) > 1 else args.eras, dfdataset.class_sample_map, regex=f"*{dfdataset.filepostfix}")

    main(dfdataset, model, filepaths, parallel=args.submission == 'parallel')
