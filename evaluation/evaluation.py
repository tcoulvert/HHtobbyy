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
    get_Dataframe, get_DMatrix, get_filepaths_func
)
from evaluation_utils import (
    get_ttH_score, get_QCD_score
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
    "--base_filepath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `get_filepaths_lambda.txt` file"
)
parser.add_argument(
    "--fold", 
    default=None,
    help="Only evaluate a specific fold"
)
parser.add_argument(
    "--train", 
    action="store_true",
    help="Evaluate and save out train"
)
parser.add_argument(
    "--test", 
    action="store_true",
    help="Evaluate and save out test"
)

################################


gpustat.print_gpustat()

################################


def evaluate(training_dirpath: str, base_filepath: str, fold: int=None, dataset: str='test'):
    class_sample_map_filepath = os.path.join(base_filepath, "class_sample_map.json")
    with open(class_sample_map_filepath, "r") as f:
        class_sample_map = json.load(f)
    formatted_order = [''.join(class_name.split(' ')) for class_name in class_sample_map.keys()]
    
    training_dirpath = os.path.join(training_dirpath, "")
    param_filepath = os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_best_params.json")
    with open(param_filepath, 'r') as f:
        param = json.load(f)
    param = list(param.items()) + [('eval_metric', 'mlogloss')]


    for fold_idx in range(5):
        if fold is not None and fold != fold_idx: continue

        booster = xgb.Booster(param)
        booster.load_model(os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_BDT_fold{fold_idx}.model"))

        filepaths = get_filepaths_func(class_sample_map, base_filepath)(fold_idx, dataset)
        for i, class_name in enumerate(filepaths.keys()):
            for filepath in filepaths[class_name]:
                df, aux = get_Dataframe(filepath), get_Dataframe(filepath, aux=True)
                aux['AUX_label1D'] = i
                dm = get_DMatrix(df, aux, dataset=dataset)

                preds = booster.predict(dm, iteration_range=(0, booster.best_iteration+1))

                for i, class_name in enumerate(formatted_order):
                    df[f"AUX_{class_name}_prob"] = preds[:, i]
                # df[f"AUX_DttH_prob"] = get_ttH_score(preds)
                # df[f"AUX_DQCD_prob"] = get_QCD_score(preds)

                for col in aux.columns:
                    df[col] = aux[col]

                df.to_parquet(filepath)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.test:
        evaluate(args.training_dirpath, args.base_filepath, args.fold)
    if args.train:
        evaluate(args.training_dirpath, args.base_filepath, args.fold, dataset='train')