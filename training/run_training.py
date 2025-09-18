# %matplotlib widget
# Stdlib packages
import datetime
import json
import os
from pathlib import Path

# HEP packages
import gpustat
import xgboost as xgb

################################


from training_utils import (
    get_filepaths_func, get_filepaths
)
from optimize_hyperparams import (
    init_params, optimize_hyperparams
)
from preprocessing.retrieval_utils import (
    get_DMatrices
)

################################


gpustat.print_gpustat()

LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/training_parquets/"
PARQUET_TIME = "2025-09-16_15-49-17"
BASE_FILEPATH = os.path.join(LPC_FILEPREFIX, PARQUET_TIME, "")

CURRENT_DIRPATH = str(Path().absolute())
VERSION = 'v16'
VARS = '22to24v2'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

OUTPUT_DIRPATH = os.path.join(CURRENT_DIRPATH, f"../MultiClassBDT_model_outputs", VERSION, VARS, CURRENT_TIME)
if not os.path.exists(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

OPTIMIZE_SPACE = False
N_CLASSES = len(get_filepaths(BASE_FILEPATH, 0, ''))
N_FOLDS = 5

################################


param_filepath = os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_best_params.json')
if OPTIMIZE_SPACE:
    print('OPTIMIZING SPACE')
    
    param, num_trees = optimize_hyperparams(
        get_filepaths_func(BASE_FILEPATH), param_filepath, verbose=True
    )
else:
    param, num_trees = init_params(N_CLASSES)
param['eval_metric'] = 'merror'
param = list(param.items()) + [('eval_metric', 'mlogloss')]

evals_result_dict = {f"fold_{fold_idx}": dict() for fold_idx in range(N_FOLDS)}
for fold_idx in range(N_FOLDS):
    print(f"fold {fold_idx}")

    train_dm, val_dm, test_dm = get_DMatrices(
        get_filepaths_func(BASE_FILEPATH), fold_idx
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