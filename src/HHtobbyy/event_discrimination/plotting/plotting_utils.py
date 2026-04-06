# Stdlib packages
import copy
import os

# Common Py packages
import numpy as np

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.evaluation import transform_preds

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
    model: Model, discriminator: str, plot_type: str, plot_func, 
    project_1D: bool=False
):
    plot_dirpath = make_plot_dirpath(model.modelconfig.output_dirpath, plot_type)

    weights = model.dfdataset.get_all_test().loc[:, f"{model.dfdataset.aux_var_prefix}eventWeight"].to_numpy()
    labels = model.dfdataset.get_all_test().loc[:, f"{model.dfdataset.aux_var_prefix}label1D"].to_numpy()
    samples = model.dfdataset.get_all_test().loc[:, f"{model.dfdataset.aux_var_prefix}sample_name"].to_numpy()

    predictions = model.evaluate_all_folds()
    transformed_labels, transformed_predictions, _ = transform_preds(model.dfdataset.class_sample_map.keys(), discriminator, predictions)

    























    plot_data = {}
    for fold_idx in range(get_n_folds(dataset_dirpath)):
        booster = get_booster(fold_idx)

        fold_plot_data =  {}

        if dataset == "train-test": 
            dm = get_train_DMatrices(dataset_dirpath, fold_idx, dataset='test')
        elif dataset == "train":
            dm = get_train_DMatrices(dataset_dirpath, fold_idx, dataset='train')
        else:
            dm = get_test_subset_DMatrix(dataset_dirpath, fold_idx, dataset.split('&'))

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
