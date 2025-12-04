# %matplotlib widget
# Stdlib packages
import json
import os
import subprocess
import sys

# Common Py packages
import numpy as np

# HEP packages
try:
    import gpustat
except ImportError:
    pass
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


def init_params(n_classes: int, static_params_dict: dict=None):
    param = {}
    # Booster parameters
    param['eta']              = 0.05       # learning rate
    param['max_depth']        = 10         # max number of splittings per tree
    param['subsample']        = 0.2        # fraction of events to train tree on
    param['colsample_bytree'] = 0.6        # fraction of features to train tree on
    param['num_class']        = n_classes  # num classes for multi-class training
    param['min_child_weight'] = 0.25
    try:
        gpustat.print_gpustat()
        param['device']           = 'cuda'
        param['tree_method']      = 'gpu_hist'
        param['sampling_method']  = 'gradient_based'
    except:
        param['device']           = 'cpu'
        param['tree_method']      = 'hist'
        param['sampling_method']  = 'uniform'
    param['max_bin']          = 512
    param['grow_policy']      = 'lossguide'
    # Learning task parameters
    param['objective']   = 'multi:softprob'   # objective function
    param['eval_metric'] = 'mlogloss'         # evaluation metric for cross validation

    if static_params_dict is not None:
        for key, value in static_params_dict.items():
            param[key] = value

    return param, round(25 / param['eta'])  # number of trees to make


def optimize_hyperparams(
    dataset_dirpath: str, n_classes: int,
    param_filepath: str, static_params_dict: dict=None,
    verbose: bool=False, verbose_eval: bool=False, start_point: int=0,
):
    # order and grouping of optimization taken from: 
    #   https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/#:~:text=min_child_weight%20%3D%201%3A%20A%20smaller%20value,%2C%20anyways%2C%20be%20tuned%20later.
    rng = np.random.default_rng()
    param, num_trees = init_params(n_classes, static_params_dict=static_params_dict)
    print("Baseline parameters: {}".format(param))

    score_arrs = {
        'max_depth_and_min_child_weight': list(),
        'min_split_loss': list(),
        'subsample_and_colsample_bytree': list(),
        'reg_lambda': list(),
        'eta': list()
    }

## max_depth and min_child_weight ##
    max_depth_range = (2, 6)
    min_child_weight_range = (0.01, 2.)
    max_depth_and_min_child_weight_space  = [
        Integer(max_depth_range[0], max_depth_range[1], "uniform", name='max_depth'),
        Real(min_child_weight_range[0], min_child_weight_range[1], "log-uniform", name='min_child_weight'),
    ]
    if 'max_depth' in static_params_dict.keys() and 'min_child_weight' in static_params_dict.keys():
        max_depth_and_min_child_weight_space = []
    elif 'max_depth' in static_params_dict.keys():
        max_depth_and_min_child_weight_space  = [
            Real(min_child_weight_range[0], min_child_weight_range[1], "log-uniform", name='min_child_weight'),
        ]
    elif 'min_child_weight' in static_params_dict.keys():
        max_depth_and_min_child_weight_space  = [
            Integer(max_depth_range[0], max_depth_range[1], "uniform", name='max_depth'),
        ]
    @use_named_args(max_depth_and_min_child_weight_space)
    def max_depth_and_min_child_weight_objective(**X):
        if verbose:
            print("New configuration: {}".format(X))

        for key, val in X.items():
            param[key] = val

        # randomly sample a fold to evaluate
        fold_idx = rng.integers(0, 4)
        train_dm, val_dm, _ = get_train_DMatrices(dataset_dirpath, fold_idx)

        booster = xgb.train(
            param, train_dm, num_boost_round=num_trees, 
            evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=verbose_eval,
        )
        eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

        best_mlogloss = float(eval_str[eval_str.find('val-mlogloss:')+len('val-mlogloss:'):])
        score_arrs['max_depth_and_min_child_weight'].append(best_mlogloss)

        if verbose:
            print(f"Best val. mlogloss on fold{fold_idx} = {best_mlogloss}")

        return -best_mlogloss
    
    if start_point == 0 and len(max_depth_and_min_child_weight_space) > 0:
        print("Optimizing max_depth (max depth of tree) and min_child_weight (min sum of weights in final nodes)")
        result_max_depth_and_min_child_weight = gp_minimize(
            max_depth_and_min_child_weight_objective, max_depth_and_min_child_weight_space,
            n_calls=10, n_points=1
        )

        if len(max_depth_and_min_child_weight_space) == 2:
            param['max_depth'] = int(result_max_depth_and_min_child_weight.x[0])
            param['min_child_weight'] = float(result_max_depth_and_min_child_weight.x[1])
        elif 'max_depth' not in static_params_dict.keys():
            param['max_depth'] = int(result_max_depth_and_min_child_weight.x[0])
        elif 'min_child_weight' not in static_params_dict.keys():
            param['min_child_weight'] = int(result_max_depth_and_min_child_weight.x[0])

        print(f"Best max_depth = {param['max_depth']} and min_child_weight = {param['min_child_weight']}")

        with open(param_filepath, 'w') as f:
            json.dump(param, f)

## min_split_loss ##
    min_split_loss_range = (0., 0.5)
    min_split_loss_space  = [
        Real(min_split_loss_range[0], min_split_loss_range[1], "uniform", name='min_split_loss'),
    ]
    @use_named_args(min_split_loss_space)
    def min_split_loss_objective(**X):
        if verbose:
            print("New configuration: {}".format(X))

        for key, val in X.items():
            param[key] = val

        # randomly sample a fold to evaluate
        fold_idx = rng.integers(0, 4)
        train_dm, val_dm, _ = get_train_DMatrices(dataset_dirpath, fold_idx)

        booster = xgb.train(
            param, train_dm, num_boost_round=num_trees, 
            evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=verbose_eval,
        )
        eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

        best_mlogloss = float(eval_str[eval_str.find('val-mlogloss:')+len('val-mlogloss:'):])
        score_arrs['min_split_loss'].append(best_mlogloss)

        if verbose:
            print(f"Best val. mlogloss on fold{fold_idx} = {best_mlogloss}")

        return -best_mlogloss

    if start_point <= 1 and 'min_split_loss' not in static_params_dict.keys():
        if start_point > 0:
            with open(param_filepath, 'r') as f:
                param = json.load(f)
            
        print("Optimizing min_split_loss (min loss change to add leaf)")
        result_min_split_loss = gp_minimize(min_split_loss_objective, min_split_loss_space)
        param['min_split_loss'] = float(result_min_split_loss.x[0])
        print(f"Best min_split_loss = {param['min_split_loss']}")

        with open(param_filepath, 'w') as f:
            json.dump(param, f)

## subsample and colsample_by_tree ##
    subsample_range = (0.3, 0.6)
    colsample_by_tree_range = (0.3, 0.9)
    subsample_and_colsample_bytree_space  = [
        Real(subsample_range[0], subsample_range[1], "log-uniform", name='subsample'),
        Real(colsample_by_tree_range[0], colsample_by_tree_range[1], "uniform", name='colsample_bytree'),
    ]
    @use_named_args(subsample_and_colsample_bytree_space)
    def subsample_and_colsample_bytree_objective(**X):
        if verbose:
            print("New configuration: {}".format(X))

        for key, val in X.items():
            param[key] = val

        # randomly sample a fold to evaluate
        fold_idx = rng.integers(0, 4)
        train_dm, val_dm, _ = get_train_DMatrices(dataset_dirpath, fold_idx)

        booster = xgb.train(
            param, train_dm, num_boost_round=num_trees, 
            evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=verbose_eval,
        )
        eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

        best_mlogloss = float(eval_str[eval_str.find('val-mlogloss:')+len('val-mlogloss:'):])
        score_arrs['subsample_and_colsample_bytree'].append(best_mlogloss)

        if verbose:
            print(f"Best val. mlogloss on fold{fold_idx} = {best_mlogloss}")

        return -best_mlogloss

    if start_point <= 2:
        if start_point > 0:
            with open(param_filepath, 'r') as f:
                param = json.load(f)
                
        print("Optimizing subsample (fraction of training events) and colsample_bytree (fraction of training features per tree)")
        result_subsample_and_colsample_bytree = gp_minimize(subsample_and_colsample_bytree_objective, subsample_and_colsample_bytree_space)
        param['subsample'] = float(result_subsample_and_colsample_bytree.x[0])
        param['colsample_bytree'] = float(result_subsample_and_colsample_bytree.x[1])
        print(f"Best subsample = {param['subsample']} and colsample_bytree = {param['colsample_bytree']}")

        with open(param_filepath, 'w') as f:
            json.dump(param, f)
        

## reg_lambda ##
    lambda_range = (0.001, 0.1)
    reg_lambda_space  = [
        Real(lambda_range[0], lambda_range[1], "log-uniform", name='reg_lambda'),
    ]
    @use_named_args(reg_lambda_space)
    def reg_lambda_objective(**X):
        if verbose:
            print("New configuration: {}".format(X))

        for key, val in X.items():
            param[key] = val

        # randomly sample a fold to evaluate
        fold_idx = rng.integers(0, 4)
        train_dm, val_dm, _ = get_train_DMatrices(dataset_dirpath, fold_idx)

        booster = xgb.train(
            param, train_dm, num_boost_round=num_trees, 
            evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=verbose_eval,
        )
        eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

        best_mlogloss = float(eval_str[eval_str.find('val-mlogloss:')+len('val-mlogloss:'):])
        score_arrs['reg_lambda'].append(best_mlogloss)

        if verbose:
            print(f"Best val. mlogloss on fold{fold_idx} = {best_mlogloss}")

        return -best_mlogloss

    if start_point <= 3:
        if start_point > 0:
            with open(param_filepath, 'r') as f:
                param = json.load(f)

        print("Optimizing reg_lambda (L2 reg)")
        result_reg_lambda = gp_minimize(reg_lambda_objective, reg_lambda_space)
        param['reg_lambda'] = float(result_reg_lambda.x[0])
        print(f"Best reg_lambda = {param['reg_lambda']}")

        with open(param_filepath, 'w') as f:
            json.dump(param, f)

## eta ##
    eta_range = (0.01, 0.3)
    eta_space  = [
        Real(eta_range[0], eta_range[1], "log-uniform", name='eta'),
    ]
    @use_named_args(eta_space)
    def eta_objective(**X):
        if verbose:
            print("New configuration: {}".format(X))

        for key, val in X.items():
            param[key] = val
        num_trees = round(25 / X['eta'])  # number of trees to make

        # randomly sample a fold to evaluate
        fold_idx = rng.integers(0, 4)
        train_dm, val_dm, _ = get_train_DMatrices(dataset_dirpath, fold_idx)

        booster = xgb.train(
            param, train_dm, num_boost_round=num_trees, 
            evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=verbose_eval,
        )
        eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

        best_mlogloss = float(eval_str[eval_str.find('val-mlogloss:')+len('val-mlogloss:'):])
        score_arrs['eta'].append(best_mlogloss)

        if verbose:
            print(f"Best val. mlogloss on fold{fold_idx} = {best_mlogloss}")

        return -best_mlogloss

    if start_point <= 4:
        if start_point > 0:
            with open(param_filepath, 'r') as f:
                param = json.load(f)
                
        print("Optimizing eta (step size)")
        result_eta = gp_minimize(eta_objective, eta_space)
        param['eta'] = float(result_eta.x[0])
        print(f"Best eta = {param['eta']}")

        with open(param_filepath, 'w') as f:
            json.dump(param, f)

    print("Best parameters: {}".format(param))
    
    return param, num_trees