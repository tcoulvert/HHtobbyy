# Common Py packages
import numpy as np

# ML packages
import GPUtil
import xgboost as xgb

# ML packages
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelConfig
from HHtobbyy.event_discrimination.models.XGBoostBDT.XGBoostBDTDataset import XGBoostBDTDataset

################################


class XGBoostBDTConfig(ModelConfig):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset

        # Required parameters
        self.output_dirpath    = ''

        # Architecture parameters
        self.eta                  = 0.1        # learning rate
        self.max_depth            = 4          # max number of splittings per tree
        self.colsample_bytree     = 1.         # fraction of features to train tree on
        self.min_child_weight     = 1.         # smallest sum weight for leaf
        self.min_split_loss       = 0.         # smallest loss reduction for leaf
        self.patience             = 10         # number of rounds to wait without improvement before early stopping

        # Penalty parameters
        self.reg_lambda           = 1.         # L2 regularization
        self.reg_alpha            = 0.         # L1 regularization
        
        # Hardware parameters
        try:
            deviceID = GPUtil.getAvailable(order='memory')[0]
            self.device           = f'cuda:{deviceID}'
            self.sampling_method  = 'gradient_based'
            self.subsample        = 0.2        # fraction of events to train tree on
        except:
            self.device           = 'cpu'
            self.sampling_method  = 'uniform'
            self.subsample        = 0.8        # fraction of events to train tree on
        self.tree_method          = 'hist'
        self.max_bin              = 256        # number of bins for histogramming
        self.grow_policy          = 'lossguide'

        # Learning task parameters
        self.objective            = 'binary:logistic'   # objective function
        self.eval_metric          = 'logloss'           # evaluation metric for cross validation

        # Binary vs. Multiclass
        if self.dfdataset.n_classes > 2:
            self.objective        = 'multi:softprob'
            self.eval_metric      = 'mlogloss'
            self.num_class        = self.dfdataset.n_classes  # num classes for multi-class training

        # Quality of life parameters
        self.verbose_eval         = 25          # Number of trees between print statements

        # Safety parameters
        self.num_boost_round      = 500         # max number of trees to make

        self.process_config(config)

    def optimize_params(self, model_dataset: XGBoostBDTDataset, static_params: dict={}, verbose: bool=False):
        # order and grouping of optimization taken from: 
        #   https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/#:~:text=min_child_weight%20%3D%201%3A%20A%20smaller%20value,%2C%20anyways%2C%20be%20tuned%20later.
        rng = np.random.default_rng(seed=self.dfdataset.seed)
        param = {key: value for key, value in self.__dict__.items()}

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
        if 'max_depth' in static_params.keys() and 'min_child_weight' in static_params.keys():
            max_depth_and_min_child_weight_space = []
        elif 'max_depth' in static_params.keys():
            max_depth_and_min_child_weight_space  = [
                Real(min_child_weight_range[0], min_child_weight_range[1], "log-uniform", name='min_child_weight'),
            ]
        elif 'min_child_weight' in static_params.keys():
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
            fold_idxs = rng.integers(0, self.dfdataset.n_folds, size=2)
            train_dm, val_dm = model_dataset.get_val(fold_idxs[0]), model_dataset.get_val(fold_idxs[1])

            booster = xgb.train(
                param, val_dm, num_boost_round=self.num_boost_round, 
                evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=self.verbose_eval,
            )
            eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

            best_eval = float(eval_str[eval_str.find(f'val-{self.eval_metric}:')+len(f'val-{self.eval_metric}:'):])
            score_arrs['max_depth_and_min_child_weight'].append(best_eval)

            if verbose:
                print(f"Best val. {self.eval_metric} on fold{fold_idxs[0]} = {best_eval}")

            return -best_eval
        
        if len(max_depth_and_min_child_weight_space) > 0:
            print("Optimizing max_depth (max depth of tree) and min_child_weight (min sum of weights in final nodes)")
            result_max_depth_and_min_child_weight = gp_minimize(
                max_depth_and_min_child_weight_objective, max_depth_and_min_child_weight_space,
                n_calls=10, n_points=1
            )

            if len(max_depth_and_min_child_weight_space) == 2:
                param['max_depth'] = int(result_max_depth_and_min_child_weight.x[0])
                param['min_child_weight'] = float(result_max_depth_and_min_child_weight.x[1])
            elif 'max_depth' not in static_params.keys():
                param['max_depth'] = int(result_max_depth_and_min_child_weight.x[0])
            elif 'min_child_weight' not in static_params.keys():
                param['min_child_weight'] = int(result_max_depth_and_min_child_weight.x[0])

            print(f"Best max_depth = {param['max_depth']} and min_child_weight = {param['min_child_weight']}")

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
            fold_idxs = rng.integers(0, self.dfdataset.n_folds, size=2)
            train_dm, val_dm = model_dataset.get_val(fold_idxs[0]), model_dataset.get_val(fold_idxs[1])

            booster = xgb.train(
                param, train_dm, num_boost_round=self.num_boost_round, 
                evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=self.verbose_eval,
            )
            eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

            best_eval = float(eval_str[eval_str.find(f'val-{self.eval_metric}:')+len(f'val-{self.eval_metric}:'):])
            score_arrs['min_split_loss'].append(best_eval)

            if verbose:
                print(f"Best val. {self.eval_metric} on fold{fold_idxs[0]} = {best_eval}")

            return -best_eval

        if 'min_split_loss' not in static_params.keys():
            print("Optimizing min_split_loss (min loss change to add leaf)")
            result_min_split_loss = gp_minimize(min_split_loss_objective, min_split_loss_space)
            param['min_split_loss'] = float(result_min_split_loss.x[0])
            print(f"Best min_split_loss = {param['min_split_loss']}")

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
            fold_idxs = rng.integers(0, self.dfdataset.n_folds, size=2)
            train_dm, val_dm = model_dataset.get_val(fold_idxs[0]), model_dataset.get_val(fold_idxs[1])

            booster = xgb.train(
                param, train_dm, num_boost_round=self.num_boost_round, 
                evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=self.verbose_eval,
            )
            eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

            best_eval = float(eval_str[eval_str.find(f'val-{self.eval_metric}:')+len(f'val-{self.eval_metric}:'):])
            score_arrs['subsample_and_colsample_bytree'].append(best_eval)

            if verbose:
                print(f"Best val. {self.eval_metric} on fold{fold_idxs[0]} = {best_eval}")

            return -best_eval
        
        print("Optimizing subsample (fraction of training events) and colsample_bytree (fraction of training features per tree)")
        result_subsample_and_colsample_bytree = gp_minimize(subsample_and_colsample_bytree_objective, subsample_and_colsample_bytree_space)
        param['subsample'] = float(result_subsample_and_colsample_bytree.x[0])
        param['colsample_bytree'] = float(result_subsample_and_colsample_bytree.x[1])
        print(f"Best subsample = {param['subsample']} and colsample_bytree = {param['colsample_bytree']}")

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
            fold_idxs = rng.integers(0, self.dfdataset.n_folds, size=2)
            train_dm, val_dm = model_dataset.get_val(fold_idxs[0]), model_dataset.get_val(fold_idxs[1])

            booster = xgb.train(
                param, train_dm, num_boost_round=self.num_boost_round, 
                evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=self.verbose_eval,
            )
            eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

            best_eval = float(eval_str[eval_str.find(f'val-{self.eval_metric}:')+len(f'val-{self.eval_metric}:'):])
            score_arrs['reg_lambda'].append(best_eval)

            if verbose:
                print(f"Best val. {self.eval_metric} on fold{fold_idxs[0]} = {best_eval}")

            return -best_eval
        
        print("Optimizing reg_lambda (L2 reg)")
        result_reg_lambda = gp_minimize(reg_lambda_objective, reg_lambda_space)
        param['reg_lambda'] = float(result_reg_lambda.x[0])
        print(f"Best reg_lambda = {param['reg_lambda']}")

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
            num_boost_round = round(50 / X['eta'])  # number of trees to make

            # randomly sample a fold to evaluate
            fold_idxs = rng.integers(0, self.dfdataset.n_folds, size=2)
            train_dm, val_dm = model_dataset.get_val(fold_idxs[0]), model_dataset.get_val(fold_idxs[1])

            booster = xgb.train(
                param, train_dm, num_boost_round=num_boost_round, 
                evals=[(train_dm, 'train'), (val_dm, 'val')], early_stopping_rounds=10, verbose_eval=self.verbose_eval,
            )
            eval_str = booster.eval(val_dm, name='val', iteration=booster.best_iteration)

            best_eval = float(eval_str[eval_str.find(f'val-{self.eval_metric}:')+len(f'val-{self.eval_metric}:'):])
            score_arrs['eta'].append(best_eval)

            if verbose:
                print(f"Best val. {self.eval_metric} on fold{fold_idxs[0]} = {best_eval}")

            return -best_eval
        
        print("Optimizing eta (step size)")
        result_eta = gp_minimize(eta_objective, eta_space)
        param['eta'] = float(result_eta.x[0])
        print(f"Best eta = {param['eta']}")

        print("Best parameters: {}".format(param))
        
        self.process_config(param, force=True)
        