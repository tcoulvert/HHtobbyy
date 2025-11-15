# Stdlib packages
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# HEP packages
import gpustat
import pyarrow.parquet as pq
import xgboost as xgb

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
    get_class_sample_map,
)
from training_utils import (
    get_dataset_filepath, get_model_func
)
from evaluation_utils import (
    get_filepaths, evaluate_and_save, 
)

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "--training_dirpath", 
    default=CWD,
    help="Full filepath on LPC for trained model files"
)
parser.add_argument(
    "--dataset_filepath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_filepath.txt` file"
)
parser.add_argument(
    "--dataset", 
    choices=["train", "test", "all"], 
    default="test",
    help="Evaluate and save out train"
)
parser.add_argument(
    "--syst_name", 
    choices=["nominal", "all"], 
    default="nominal",
    help="Evaluate and save out train"
)

################################


gpustat.print_gpustat()

################################


def evaluate_model(training_dirpath: str, dataset_filepath: str, dataset: str="test", syst_name="nominal"):
    class_sample_map = get_class_sample_map(dataset_filepath)
    formatted_classes = [''.join(class_name.split(' ')) for class_name in class_sample_map.keys()]
    
    get_booster = get_model_func(training_dirpath)

    for fold_idx in range(5):

        booster = get_booster(fold_idx)

        filepaths = get_filepaths(dataset_filepath, dataset, syst_name)(fold_idx)

        for i, class_name in enumerate(filepaths.keys()):
            for filepath in filepaths[class_name]:

                evaluate_and_save(filepath, booster, formatted_classes)

                

if __name__ == "__main__":
    args = parser.parse_args()

    training_dirpath = os.path.join(args.training_dirpath, "")
    if args.dataset_filepath is None:
        dataset_filepath = get_dataset_filepath(args.training_filepath)
    else:
        dataset_filepath = args.dataset_filepath
    
    evaluate_model(training_dirpath, dataset_filepath, args.dataset, args.syst_name)