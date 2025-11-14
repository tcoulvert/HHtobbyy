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
# PARQUET_TIME = "2025-10-22_10-35-14"  # 2022-23 PNet
# PARQUET_TIME = "2025-10-22_10-10-00"  # 2022-23 WPs
# PARQUET_TIME = "2025-10-22_10-10-24"  # 2022-23 WPs w/ extra kappa lambda samples
# PARQUET_TIME = "2025-10-22_10-34-09"  # 2022-23 WPs w/ 1 large batch sample
# PARQUET_TIME = "2025-10-22_10-10-13"  # 2022-23 WPs w/ all large batch samples
# PARQUET_TIME = "2025-10-22_15-52-13"  # 2022-23 WPs w/ all large batch samples + extra kappa lambda samples
# PARQUET_TIME = "2025-11-14_12-14-01"  # 2022-23 WPs + 3XT + 4XT
# PARQUET_TIME = "2025-10-22_10-36-34"  # 2024 UParT  
# PARQUET_TIME = "2025-10-22_10-36-51"  # 2024 WPs
# PARQUET_TIME = "2025-10-24_20-33-24"  # 2024 WPs + 3XT + 4XT
# PARQUET_TIME = "2025-10-23_02-31-07"  # 2024 PNet
# PARQUET_TIME = "2025-10-21_11-17-26"  # 2022-24 WPs
# PARQUET_TIME = "2025-11-11_14-13-23"  # 2022-24 WPs + high stats
# PARQUET_TIME = "2025-11-14_13-28-53"  # 2022-24 WPs + high stats + 3XT + 4XT -- USE THIS ONE
# PARQUET_TIME = "2025-11-14_13-31-51"  # 2022-24 WPs + high stats + 3XT + 4XT + MHH
BASE_FILEPATH = os.path.join(LPC_FILEPREFIX, PARQUET_TIME, "")

CURRENT_DIRPATH = str(Path().absolute())
VERSION = 'v17'
VARS = '22to24_bTagWPbatch3XT4XT'
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