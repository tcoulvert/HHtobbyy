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

    # Create cache directory
    dataset_dirpath = os.path.normpath(dataset_dirpath)
    cache_dir = os.path.join(training_dirpath, "cached_eval", os.path.basename(dataset_dirpath))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    force_refresh = False  # Whether to force re-evaluation even if cached data exists
    restandardize = True
    if restandardize:
        # use the standardization json files in the training dir
        post_std_filepath = os.path.join(training_dirpath, "standardization.json")
        pre_std_filepath = os.path.join(dataset_dirpath, "standardization.json")
        if not os.path.exists(pre_std_filepath) or not os.path.exists(post_std_filepath):
            restandardize = False

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, discriminator)

    plot_data = {
        class_name: {'preds': None, 'labels': None, 'weights': None}
        for class_name in CLASS_NAMES
    }

    for fold_idx in range(get_n_folds(dataset_dirpath)):
        booster = get_booster(fold_idx)

        fold_plot_data =  {
            class_name: {'preds': None, 'labels': None, 'weights': None}
            for class_name in CLASS_NAMES
        }

        # Check for cached data
        cache_filename = f"fold{fold_idx}_{dataset}.npz"
        cache_filepath = os.path.join(cache_dir, cache_filename)
        
        preds_all = None
        labels_all = None
        weights_all = None
        name_all = None

        if os.path.exists(cache_filepath):
            print(f"[INFO] Loading cached evaluation from {cache_filepath}")
            try:
                cached_data = np.load(cache_filepath, allow_pickle=True)
                preds_all = cached_data['preds']
                labels_all = cached_data['labels']
                weights_all = cached_data['weights']
                name_all = cached_data['name']
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}")
        
        if preds_all is None or force_refresh:
            print(f"[INFO] Running evaluation for fold {fold_idx}, dataset {dataset}")
            retrieval_kwargs = {'plot_mode': True}
            # get training parameters from training dir name
            extension_dict = {
                'using_resolution_var': '_with_resolution',
                'do_reweight': '_reweighted',
                'do_mass_cut': '_masscut',
            }
            model_dir = os.path.basename(os.path.normpath(training_dirpath))
            trainging_param = {key: value in model_dir for key, value in extension_dict.items()}

            retrieval_kwargs.update(trainging_param)
            if dataset == "train-test":
                retrieval_kwargs.update({'test_only': True})
            retrieval_kwargs.update({'get_aux': True})
            if restandardize:
                retrieval_kwargs.update({
                    'restandardize': True,
                    'previous_std': pre_std_filepath,
                    'new_std': post_std_filepath,
                })
            train_dm, _, test_dm, train_aux, _, test_aux = get_train_DMatrices(dataset_dirpath, fold_idx, **retrieval_kwargs)

            if dataset == "train-test": dm = test_dm
            elif dataset == "train": dm = train_dm
            else: raise ValueError(f"Unknown dataset: {dataset}")

            preds_all = evaluate(booster, dm)
            labels_all = dm.get_label()
            weights_all = dm.get_weight()
            name_all = test_aux['AUX_sample_name'] if dataset == "train-test" else train_aux['AUX_sample_name']
            
            print(f"[INFO] Saving evaluation to {cache_filepath}")
            np.savez(cache_filepath, preds=preds_all, labels=labels_all, weights=weights_all, name=name_all)

        for j, class_name in enumerate(CLASS_NAMES):
            event_mask = (labels_all == j)

            nD_preds = preds_all[event_mask]
            transformed_preds = transform_preds(nD_preds)

            # Add preds to full list for cross-fold evaluation
            fold_plot_data[class_name]['preds'] = transformed_preds
            fold_plot_data[class_name]['labels'] = labels_all[event_mask]
            fold_plot_data[class_name]['weights'] = weights_all[event_mask]
            for data_name, data in fold_plot_data[class_name].items():
                if plot_data[class_name][data_name] is None:
                    plot_data[class_name][data_name] = copy.deepcopy(data)
                else:
                    plot_data[class_name][data_name] = np.concatenate([plot_data[class_name][data_name], data])
        if get_n_folds(dataset_dirpath) > 1:
            if not project_1D: plot_func(fold_plot_data, transform_labels, plot_dirpath, plot_postfix=f'fold{fold_idx}')
            else: project_1D_output(fold_plot_data, transform_labels, plot_dirpath, plot_func, plot_postfix=f'fold{fold_idx}')
        
    if not project_1D: plot_func(plot_data, transform_labels, plot_dirpath)
    else: project_1D_output(plot_data, transform_labels, plot_dirpath, plot_func)
