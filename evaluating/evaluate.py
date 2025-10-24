# %matplotlib widget
# Stdlib packages
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# HEP packages
import gpustat
import xgboost as xgb

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))


import training_utils as utils
from retrieval_utils import (
    get_DMatrices
)

################################


gpustat.print_gpustat()

order = ['ggF HH', 'ttH + bbH', 'VH', 'non-res + ggFH + VBFH']

################################


def evaluate_model(output_dirpath: str, base_filepath: str, fold: int=None, preds: str='train-val-test'):
    output_dirpath = os.path.join(output_dirpath, "")
    param_filepath = os.path.join(output_dirpath, f'{output_dirpath.split('/')[-2]}_best_params.json')
    with open(param_filepath, 'r') as f:
        param = json.load(f)
    param = list(param.items()) + [('eval_metric', 'mlogloss')]

    train_preds, val_preds, test_preds = {}, {}, {}
    for fold_idx in range(5):
        if fold is not None and fold != fold_idx: continue
        print(f"fold {fold_idx}")

        booster = xgb.Booster(param)
        booster.load_model(os.path.join(output_dirpath, f"{output_dirpath.split('/')[-2]}_BDT_fold{fold_idx}.model"))

        train_dm, val_dm, test_dm = get_DMatrices(
            utils.get_filepaths_func(base_filepath), fold_idx
        )

        if 'train' in preds.lower():
            train_preds[f"fold_{fold_idx}"] = booster.predict(
                train_dm, iteration_range=(0, booster.best_iteration+1)
            )
        if 'val' in preds.lower():
            val_preds[f"fold_{fold_idx}"] = booster.predict(
                val_dm, iteration_range=(0, booster.best_iteration+1)
            )
        if 'test' in preds.lower():
            test_preds[f"fold_{fold_idx}"] = booster.predict(
                test_dm, iteration_range=(0, booster.best_iteration+1)
            )

    return_tuple = tuple([preds_dict for preds_name, preds_dict in zip(['train', 'val', 'test'], [train_preds, val_preds, test_preds]) if preds_name in preds.lower()])
    return return_tuple

