# Stdlib packages
import json
import os
import subprocess
import sys

# Common Py packages
import numpy as np

# HEP packages
import xgboost as xgb

# ML packages
from sklearn.metrics import log_loss

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))

from retrieval_utils import get_labelND

################################


def get_dataset_filepath(training_dirpath: str):
    dataset_filepath_filepath = os.path.join(training_dirpath, "dataset_filepath.txt")
    with open (dataset_filepath_filepath, "r") as f:
        dataset_filepath = f.read().strip()
    return dataset_filepath

def get_param(training_dirpath: str):
    param_filepath = os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_best_params.json")
    with open(param_filepath, 'r') as f:
        param = json.load(f)
    param = list(param.items()) + [('eval_metric', 'mlogloss')]
    return param

def get_model_func(training_dirpath: str):
    booster = xgb.Booster(get_param(training_dirpath))
    return lambda fold_idx: booster.load_model(os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_BDT_fold{fold_idx}.model"))

def mlogloss_binlogloss(
    predt: np.ndarray, dtrain: xgb.DMatrix, mLL=True, **kwargs
):
    # assert (len(kwargs) == 0 and mLL) or len(kwargs) == (n_classes - 1)
    assert (len(kwargs) == 0 and mLL)

    mweight = dtrain.get_weight()
    monehot = get_labelND(dtrain.get_label())
    mlogloss = log_loss(monehot, predt, sample_weight=mweight, normalize=False)

    bkgloglosses = {}
    for i, (key, value) in enumerate(kwargs.items(), start=1):
        bkgbool = np.logical_or(mweight == 0, mweight == i)
        bkgloglosses[key] = value * log_loss(
            monehot[bkgbool], predt[bkgbool, 0] / (predt[bkgbool, 0] + predt[bkgbool, i]),
            sample_weight=mweight[bkgbool], normalize=False
        )

    if len(bkgloglosses) > 0 and mLL:
        return f'mLL+binLL@{bkgloglosses.keys()}', float(np.sum([mlogloss]+list(bkgloglosses.values())))
    elif len(bkgloglosses) > 0:
        return f'binLL@{bkgloglosses.keys()}', float(np.sum(bkgloglosses.values()))
    else:
        return 'mLL', float(mlogloss)

def thresholded_weighted_merror(predt: np.ndarray, dtrain: xgb.DMatrix, threshold=0.95):
    """Used when there's no custom objective."""
    # No need to do transform, XGBoost handles it internally.
    weights = dtrain.get_weight()
    thresh_weight_merror = np.where(
        np.logical_and(
            np.max(predt, axis=1) >= threshold,
            np.argmax(predt, axis=1) == dtrain.get_label()
        ),
        0,
        weights
    )
    return f'WeightedMError@{threshold:.2f}', np.sum(thresh_weight_merror)