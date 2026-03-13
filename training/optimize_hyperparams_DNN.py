# %matplotlib widget
# Stdlib packages
import json
import os
import subprocess
import sys

# Common Py packages
import numpy as np

# HEP packages
import gpustat
import xgboost as xgb

# ML packages
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))

from retrieval_utils import get_train_DMatrices

################################


def init_params(static_params_dict: dict=None):
    param = {}
    # DNN parameters
    param['batch_size']    = 2048    # learning rate -- 0.05
    param['num_layers']    = 4       # number of hidden layers
    param['num_nodes']     = 50      # dimensionality of hidden layers
    param['dropout_prob']  = 0.3     # probability of dropping connections

    # Eary stopping parameters
    param['min_delta']     = 0.      # smallest val_loss difference
    param['patience']      = 4       # number of epochs to wait before early stopping

    # Dataloader parameters
    param['num_workers']   = 1       # number of workers to load data

    # Hardware parameters
    param['num_gpus']      = 1
    param['strategy']      = 'gpu'
    param['num_nodes']     = 1
    param['num_processes'] = 1

    # Safety parameters
    param['max_epochs']   = 10      # max number of epochs to run

    if static_params_dict is not None:
        for key, value in static_params_dict.items():
            param[key] = value

    return param


def optimize_hyperparams(
    dataset_dirpath: str, n_classes: int,
    param_filepath: str, static_params_dict: dict=None,
    verbose: bool=False, verbose_eval: bool=False, start_point: int=0,
):
    pass