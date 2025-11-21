# Stdlib packages
import os
import subprocess
import sys
from pathlib import Path

# Common Py packages
import numpy as np
from matplotlib import pyplot as plt

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
    get_class_sample_map, get_n_folds, get_train_DMatrices
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


def project_1D_output(
    plot_data: dict, transform_labels: list, plot_dirpath: str, 
    plot_func: function,
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
        plot_func(
            plot_data_1D, plot_dirpath, plot_prefix=f"{plot_prefix + ('_' if plot_prefix != '' else '')}{pred_label}", plot_postfix=plot_postfix
        )

def make_plot_data(
    training_dirpath: str, dataset_dirpath: str, 
    dataset: str, discriminator: str, plot_type: str,
    plot_func: function, project_1D: bool=False
):
    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)

    get_booster = get_model_func(training_dirpath)
    CLASS_SAMPLE_MAP = get_class_sample_map(dataset_dirpath)
    CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, discriminator)

    plot_data = {
        class_name: {'preds': np.array([]), 'labels': np.array([]), 'weights': np.array([])}
        for class_name in CLASS_NAMES
    }

    for fold_idx in range(get_n_folds(dataset_dirpath)):
        booster = get_booster(fold_idx)

        fold_plot_data =  {
            class_name: {'preds': None, 'labels': None, 'weights': None}
            for class_name in CLASS_NAMES
        }

        for j, class_name in enumerate(CLASS_NAMES):
            train_dm, _, test_dm = get_train_DMatrices(dataset_dirpath, fold_idx)

            if dataset == "train-test": dm = test_dm
            elif dataset == "train": dm = train_dm

            event_mask = (dm.get_label() == j)

            nD_preds = evaluate(booster, dm)[event_mask]
            transformed_preds = transform_preds(nD_preds)

            # Add preds to full list for cross-fold evaluation
            fold_plot_data[class_name]['preds'] = transformed_preds
            fold_plot_data[class_name]['labels'] = dm.get_label()[event_mask]
            fold_plot_data[class_name]['weights'] = dm.get_weight()[event_mask]
            for data_name, data in fold_plot_data[class_name].items():
                plot_data[class_name][data_name] = np.concatenate(plot_data[class_name][data_name], data)

        if not project_1D: plot_func(fold_plot_data, transform_labels, plot_dirpath, plot_postfix=f'fold{fold_idx}')
        else: project_1D_output(fold_plot_data, transform_labels, plot_dirpath, plot_func, plot_postfix=f'fold{fold_idx}')
        
    if not project_1D: plot_func(plot_data, transform_labels, plot_dirpath)
    else: project_1D_output(plot_data, transform_labels, plot_dirpath, plot_func)
