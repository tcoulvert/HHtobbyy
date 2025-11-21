# Stdlib packages
import argparse
import logging
import os
import subprocess
import sys

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
sys.path.append(os.path.join(GIT_REPO, "training/"))


from retrieval_utils import (
    get_class_sample_map, get_n_folds, format_class_names
)
from training_utils import (
    get_dataset_dirpath, get_model_func
)
from evaluation_utils import (
    get_filepaths, evaluate_and_save, 
)

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath", 
    help="Full filepath on LPC for trained model files"
)
parser.add_argument(
    "--dataset_dirpath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_dirpath.txt` file"
)
parser.add_argument(
    "--dataset", 
    choices=["train", "test", "train-test", "all"], 
    default="test",
    help="Evaluate and save out evaluation for what dataset"
)
parser.add_argument(
    "--syst_name", 
    choices=["nominal", "all"], 
    default="nominal",
    help="Evaluate and save out evaluation for what systematic of a dataset"
)

################################


def evaluate_model(training_dirpath: str, dataset_dirpath: str, dataset: str="test", syst_name="nominal"):
    class_sample_map = get_class_sample_map(dataset_dirpath)
    formatted_classes = format_class_names(class_sample_map.keys())
    
    get_booster = get_model_func(training_dirpath)

    for fold_idx in range(get_n_folds(dataset_dirpath)):

        booster = get_booster(fold_idx)

        filepaths = get_filepaths(dataset_dirpath, dataset, syst_name)(fold_idx)

        for i, class_name in enumerate(filepaths.keys()):
            for filepath in filepaths[class_name]:

                evaluate_and_save(filepath, booster, formatted_classes)
                

if __name__ == "__main__":
    args = parser.parse_args()

    training_dirpath = os.path.join(args.training_dirpath, "")
    if args.dataset_dirpath is None:
        dataset_dirpath = get_dataset_dirpath(args.training_dirpath)
    else:
        dataset_dirpath = args.dataset_dirpath
    
    evaluate_model(training_dirpath, dataset_dirpath, args.dataset, args.syst_name)