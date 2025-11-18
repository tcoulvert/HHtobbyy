# %matplotlib widget
# Stdlib packages
import datetime
import inspect
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
    get_train_DMatrices, get_class_sample_map
)

################################


gpustat.print_gpustat()

LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/training_parquets/"
# PARQUET_TIME = "2025-11-18_13-46-20"  # 2022-23 WPs + 3XT + 4XT
# PARQUET_TIME = "2025-11-18_14-54-32"  # 2022-24 WPs
# PARQUET_TIME = "2025-11-18_14-56-13"  # 2022-24 WPs + extra kl
# PARQUET_TIME = "2025-11-17_09-49-01"  # 2022-24 WPs + high stats -- USE THIS ONE
# PARQUET_TIME = "2025-11-18_13-48-48"  # 2022-24 WPs + high stats + 3XT + 4XT
# PARQUET_TIME = "2025-11-18_13-48-35"  # 2022-24 WPs + high stats + MHH
DATASET_DIRPATH = os.path.join(LPC_FILEPREFIX, PARQUET_TIME, "")

CURRENT_DIRPATH = str(Path().absolute())
VERSION = 'v18'
VARS = '22to24_bTagWPbatch'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

OPTIMIZE_SPACE = False
N_CLASSES = None
N_FOLDS = 5

################################


OUTPUT_DIRPATH = os.path.join(CURRENT_DIRPATH, f"../MultiClassBDT_model_outputs", VERSION, VARS, CURRENT_TIME)
if not os.path.exists(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

# Dict defining which samples are in what classes (see `resolved_BDT.py` for more details)
CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
N_CLASSES = len(CLASS_SAMPLE_MAP)

# txt file pointing to location of standardized dataset used for training
#  and therefore the default location for testing
dataset_filepath = os.path.join(OUTPUT_DIRPATH, "dataset_filepath.txt")
with open (dataset_filepath, "w") as f:
    f.write(DATASET_DIRPATH)

# Dict of hyperparameters for the model -- necessary to store for evaluation
param_filepath = os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_best_params.json')
if OPTIMIZE_SPACE:
    print('OPTIMIZING SPACE')
    
    param, num_trees = opt.optimize_hyperparams(
        DATASET_DIRPATH, N_CLASSES, param_filepath, verbose=True
    )
else:
    param, num_trees = opt.init_params(N_CLASSES)
param['eval_metric'] = 'merror'
with open(param_filepath, 'w') as f:
    json.dump(param, f)
param = list(param.items()) + [('eval_metric', 'mlogloss')]

# Train the model
evals_result_dict = {f"fold_{fold_idx}": dict() for fold_idx in range(N_FOLDS)}
for fold_idx in range(N_FOLDS):
    print(f"fold {fold_idx}")

    train_dm, val_dm, test_dm = get_train_DMatrices(DATASET_DIRPATH, fold_idx)

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