# Stdlib packages
import os
import subprocess
import sys

# HEP packages
import pyarrow.parquet as pq
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
    get_test_Dataframe, get_test_DMatrix,
    get_train_filepaths_func, get_test_filepaths_func
)

################################


def get_filepaths(dataset_dirpath: str, dataset: str, syst_name: str):
    if dataset == "test":
        return get_test_filepaths_func(dataset_dirpath, syst_name=syst_name)
    if dataset == "train":
        return get_train_filepaths_func(dataset_dirpath, syst_name=syst_name)
    else:
        return lambda fold_idx: get_train_filepaths_func(dataset_dirpath, syst_name=syst_name)(fold_idx) | get_test_filepaths_func(dataset_dirpath, syst_name=syst_name)(fold_idx)

def evaluate(booster: xgb.Booster, dmatrix: xgb.DMatrix):
    return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration))

def evaluate_and_save(filepath: str, booster: xgb.Booster, formatted_classes: list):
    df = get_test_Dataframe(filepath)
    dm = get_test_DMatrix(filepath)

    preds = evaluate(booster, dm)

    try:
        for i, class_name in enumerate(formatted_classes):
            df[f"AUX_{class_name}_prob"] = preds[:, i]
    except:
        print(f"Saving of evaluation of {filepath}, continuing with other samples.")

    df.to_parquet(filepath)


def get_ttH_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])

def get_QCD_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])
