# Stdlib packages
import os
import subprocess
import sys

# Common Py packages
import numpy as np

# HEP packages
import xgboost as xgb

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))


from retrieval_utils import (
    format_class_names,
    get_test_Dataframe, get_test_DMatrix,
    get_train_filepaths_func, get_test_filepaths_func
)

################################


TRANSFORM_PREDS = [
    {
        'name': 'nD', 
        'output': lambda class_names: ['D'+ formatted_class_name for formatted_class_name in format_class_names(class_names)], 
        'ROC_bkgeffs': lambda class_names: [1e-3 for _ in class_names],
        'func': lambda multibdt_output: multibdt_output
    },
    {
        'name': 'DttH-DQCD', 
        'output': lambda class_names: ['DttH', 'DQCD'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-3],
        'func': lambda multibdt_output: np.column_stack([DttH(multibdt_output), DQCD(multibdt_output)])
    },
]

################################


def get_filepaths(dataset_dirpath: str, dataset: str, syst_name: str):
    if dataset == "test":
        return get_test_filepaths_func(dataset_dirpath, syst_name=syst_name)
    elif dataset == "train":
        return get_train_filepaths_func(dataset_dirpath, syst_name=syst_name)
    elif dataset == "train-test":
        return get_train_filepaths_func(dataset_dirpath, dataset='test', syst_name=syst_name)
    else:
        return lambda fold_idx: get_train_filepaths_func(dataset_dirpath, syst_name=syst_name)(fold_idx) | get_test_filepaths_func(dataset_dirpath, syst_name=syst_name)(fold_idx)


def transform_preds_options():
    return [transformation['name'] for transformation in TRANSFORM_PREDS]

def transform_preds_bkgeffs(class_names: list, transform_name: str):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    ROC_bkgeffs = [transformation['ROC_bkgeffs'](class_names) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return ROC_bkgeffs

def transform_preds_func(class_names: list, transform_name: str):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    output, func = [(transformation['output'](class_names), transformation['func']) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return output, func

def DttH(multibdt_output):
    DttH_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])
    DttH_preds[np.isnan(DttH_preds)] = 0
    return DttH_preds

def DQCD(multibdt_output):
    DQCD_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])
    DQCD_preds[np.isnan(DQCD_preds)] = 0
    return DQCD_preds


def evaluate(booster: xgb.Booster, dmatrix: xgb.DMatrix):
    return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration))

def evaluate_and_save(filepath: str, booster: xgb.Booster, class_names: list, transform_name: str):
    transform_labels, transform_preds = transform_preds_func(class_names, transform_name)

    df = get_test_Dataframe(filepath)
    dm = get_test_DMatrix(filepath)

    preds = evaluate(booster, dm)
    transformed_preds = transform_preds(preds)

    for i, transform_label in enumerate(transform_labels):
        df[f"AUX_{transform_label}_prob"] = transformed_preds[:, i]
    
    if filepath.split('/')[1] == 'eos':
        tmp_file = f"tmp.parquet"
        df.to_parquet(tmp_file)
        subprocess.run(['eoscp', '-f', tmp_file, '/'.join(filepath.split('/')[3:])])
        subprocess.run(['rm', tmp_file])
    else:
        df.to_parquet(filepath)

    assert all(f'AUX_{transform_label}_prob' in get_test_Dataframe(filepath).columns for transform_label in transform_labels), f"{filepath.split('/')[-3:]} - evaluation and saving did not work properly"