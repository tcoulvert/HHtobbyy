# Stdlib packages
import copy
import os
import subprocess
import sys

# Common Py packages
import numpy as np

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
sys.path.append(os.path.join(GIT_REPO, "training/"))
sys.path.append(os.path.join(GIT_REPO, "evaluation/"))

# Module packages
from retrieval_utils import (
    get_class_sample_map, get_n_folds, get_train_DMatrices,
    get_test_filepaths_func, get_data_DMatrix
)
from training_utils import (
    get_model_func
)
from evaluation_utils import (
    evaluate, transform_preds_func
)

################################


def make_plot_dirpath(training_dirpath: str, plot_type: str):
    plot_dirpath = os.path.join(training_dirpath, "plots", plot_type)
    if not os.path.exists(plot_dirpath):
        os.makedirs(plot_dirpath)
    return plot_dirpath

def pad_list(list_of_lists):
    max_length = np.max([len(list_i) for list_i in list_of_lists])
    for list_i in list_of_lists:
        while len(list_i) < max_length:
            list_i.append(list_i[-1])

    return list_of_lists

def plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='png'):
    plot_prefix = plot_prefix + ('_' if plot_prefix != '' else '')
    plot_postfix = ('_' if plot_postfix != '' else '') + plot_postfix
    plot_name = plot_prefix + plot_name + plot_postfix + f'.{format}'

    plot_filepath = os.path.join(plot_dirpath, plot_name)
    return plot_filepath

def combine_prepostfix(old_plot_prepostfix: str, new_substr: str, fixtype: str):
    if new_substr == '': return old_plot_prepostfix
    if fixtype not in ['prefix', 'postfix']: raise KeyError(f"Type {fixtype} not in allowed types: {['prefix', 'postfix']}")
    elif fixtype == 'prefix':
        plot_fix = old_plot_prepostfix + ('_' if old_plot_prepostfix != '' else '') + new_substr
    elif fixtype == 'postfix':
        plot_fix = old_plot_prepostfix + ('_' if old_plot_prepostfix != '' else '') + new_substr
    return plot_fix

def float_to_str(floating_point: float):
    new_str = f"{floating_point:.2f}"
    new_str.replace('.', 'p')
    return new_str


def project_1D_output(
    plot_data: dict, transform_labels: list, plot_dirpath: str, plot_func,
    plot_prefix: str='', plot_postfix: str=''
):
    for pred_idx, pred_label in enumerate(transform_labels):
        plot_data_1D = {
            class_name: {
                'preds': class_data['preds'][:, pred_idx], 
                'labels': class_data['labels'], 
                'weights': class_data['weights']
            } for class_name, class_data in plot_data.items()
        }
        new_plot_prefix = combine_prepostfix(plot_prefix, pred_label, fixtype='prefix')
        plot_func(
            plot_data_1D, plot_dirpath, pred_idx, plot_prefix=new_plot_prefix, plot_postfix=plot_postfix
        )

def make_plot_data(
    training_dirpath: str, dataset_dirpath: str, 
    dataset: str, discriminator: str, plot_type: str, plot_func, 
    project_1D: bool=False
):
    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)

    get_booster = get_model_func(training_dirpath)
    CLASS_SAMPLE_MAP = get_class_sample_map(dataset_dirpath)
    CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, discriminator)

    plot_data = {}
    for fold_idx in range(get_n_folds(dataset_dirpath)):
        booster = get_booster(fold_idx)

        fold_plot_data =  {}

        if dataset == "train-test": 
            dm = get_train_DMatrices(dataset_dirpath, fold_idx, dataset='test')
        elif dataset == "train":
            dm = get_train_DMatrices(dataset_dirpath, fold_idx, dataset='train')
        elif dataset == "data":
            dm = get_data_DMatrix(dataset_dirpath, fold_idx)
        else: raise ValueError(f"Unknown dataset: {dataset}")

        preds = evaluate(booster, dm)
        labels = dm.get_label()
        weights = dm.get_weight()

        for j, class_name in enumerate(CLASS_NAMES):
            event_mask = (labels == j)
            if np.all(~event_mask): continue
            fold_plot_data[class_name] = {}
            if class_name not in plot_data.keys(): plot_data[class_name] = {}

            nD_preds = preds[event_mask]
            transformed_preds = transform_preds(nD_preds)

            # Add preds to full list for cross-fold evaluation
            fold_plot_data[class_name]['preds'] = transformed_preds
            fold_plot_data[class_name]['labels'] = labels[event_mask]
            fold_plot_data[class_name]['weights'] = weights[event_mask]
            for data_name, data in fold_plot_data[class_name].items():
                if data_name not in plot_data[class_name].keys():
                    plot_data[class_name][data_name] = copy.deepcopy(data)
                else:
                    plot_data[class_name][data_name] = np.concatenate([plot_data[class_name][data_name], data])

        if get_n_folds(dataset_dirpath) > 1:
            if not project_1D: plot_func(fold_plot_data, transform_labels, plot_dirpath, plot_postfix=f'fold{fold_idx}')
            else: project_1D_output(fold_plot_data, transform_labels, plot_dirpath, plot_func, plot_postfix=f'fold{fold_idx}')
        
    if not project_1D: plot_func(plot_data, transform_labels, plot_dirpath)
    else: project_1D_output(plot_data, transform_labels, plot_dirpath, plot_func)
