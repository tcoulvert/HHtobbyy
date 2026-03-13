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
    param['batch_size']        = 2048        # learning rate -- 0.05
    param['num_layers']        = 4           # number of hidden layers
    param['num_nodes']         = 50          # dimensionality of hidden layers
    param['dropout_prob']      = 0.3         # probability of dropping connections

    # Eary stopping parameters
    param['min_delta']         = 0.          # smallest val_loss difference
    param['patience']          = 4           # number of epochs to wait before early stopping
    param['monitor']           = "val_loss"  # what to track for EarlyStopping
    param['mode']              = "min"       # stop when no-longer decreasing

    # Dataloader parameters
    param['num_workers']       = 1           # number of workers to load data

    # Hardware parameters
    try:
        gpustat.print_gpustat()
        param['accelerator']   = 'gpu'       # device to use for training
    except:
        param['accelerator']   = 'cpu'
    param['strategy']          = 'auto'      # high-level how to do training
    param['num_nodes']         = 1           # number of gpu nodes to use
    param['precision']         = '32'        # 32-bit floating point
    param['gradient_clip_val'] = 10.         # max abs. value for gradient
    param['logger']            = True        # default Tensorboard logging

    # Safety parameters
    param['max_epochs']        = 500         # max number of epochs to run

    if static_params_dict is not None:
        for key, value in static_params_dict.items():
            param[key] = value

    return param


def optimize_hyperparams(
    dataset_dirpath: str, 
    param_filepath: str, static_params_dict: dict=None,
    verbose: bool=False, verbose_eval: bool=False, start_point: int=0,
):
    pass