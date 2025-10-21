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
import optimize_hyperparams as opt
from retrieval_utils import (
    get_DMatrices
)

################################


gpustat.print_gpustat()

LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/training_parquets/"
# PARQUET_TIME = "2025-10-08_14-25-00"  # 2022-23 WPs
# PARQUET_TIME = "2025-10-20_16-57-16"  # 2022-23 PNet w/ extra kappa lambda samples
# PARQUET_TIME = "2025-10-20_16-57-44"  # 2022-23 PNet w/ large batch sample
# PARQUET_TIME = "2025-10-07_17-32-07"  # 2024 WPs  
# PARQUET_TIME = "2025-10-20_16-16-12"  # 2024 PNet w/ merge fix
# PARQUET_TIME = "2025-10-09_16-32-50"  # 2024 UParT bTags
PARQUET_TIME = "2025-10-20_16-16-12"  # 2024 PNet bTags
# PARQUET_TIME = ""  # 2022-24 WPs
BASE_FILEPATH = os.path.join(LPC_FILEPREFIX, PARQUET_TIME, "")

CURRENT_DIRPATH = str(Path().absolute())
VERSION = 'v16'
VARS = '24_bTagPNet'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

OUTPUT_DIRPATH = os.path.join(CURRENT_DIRPATH, f"../MultiClassBDT_model_outputs", VERSION, VARS, CURRENT_TIME)
if not os.path.exists(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

OPTIMIZE_SPACE = False
N_CLASSES = len(utils.get_filepaths(BASE_FILEPATH, 0, ''))
N_FOLDS = 5

################################


param_filepath = os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_best_params.json')
if OPTIMIZE_SPACE:
    print('OPTIMIZING SPACE')
    
    param, num_trees = opt.optimize_hyperparams(
        utils.get_filepaths_func(BASE_FILEPATH), param_filepath, verbose=True
    )
else:
    param, num_trees = opt.init_params(N_CLASSES)
param['eval_metric'] = 'merror'
with open(param_filepath, 'w') as f:
    json.dump(param, f)
param = list(param.items()) + [('eval_metric', 'mlogloss')]

evals_result_dict = {f"fold_{fold_idx}": dict() for fold_idx in range(N_FOLDS)}
for fold_idx in range(N_FOLDS):
    print(f"fold {fold_idx}")

    train_dm, val_dm, test_dm = get_DMatrices(
        utils.get_filepaths_func(BASE_FILEPATH), fold_idx
    )

    # Train bdt
    evallist = [(train_dm, 'train'), (val_dm, 'test'), (test_dm, 'val')]
    booster = xgb.train(
        param, train_dm, num_boost_round=num_trees, 
        evals=evallist, early_stopping_rounds=10, 
        verbose_eval=25, evals_result=evals_result_dict[f"fold_{fold_idx}"],
    )

    booster.save_model(os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_BDT_fold{fold_idx}.model'))
    
    # Print perf on test dataset
    print(booster.eval(test_dm, name='test', iteration=booster.best_iteration))
    print('='*100)
            
with open(os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_BDT_eval_result.json'), 'w') as f:
    json.dump(evals_result_dict, f)