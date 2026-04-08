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


# def project_1D_output(
#     plot_data: dict, transform_labels: list, plot_dirpath: str, plot_func,
#     plot_prefix: str='', plot_postfix: str=''
# ):
#     for pred_idx, pred_label in enumerate(transform_labels):
#         plot_data_1D = {
#             class_name: {
#                 'preds': class_data['preds'][:, pred_idx], 
#                 'labels': class_data['labels'], 
#                 'weights': class_data['weights']
#             } for class_name, class_data in plot_data.items()
#         }
#         new_plot_prefix = combine_prepostfix(plot_prefix, pred_label, fixtype='prefix')
#         plot_func(
#             plot_data_1D, plot_dirpath, pred_idx, plot_prefix=new_plot_prefix, plot_postfix=plot_postfix
#         )

def make_plot_data(model: Model, discriminator: str, fold: int=-1, syst_name: str = 'nominal', regex: str | list[str] = ''):
    if fold >= 0: return make_fold_plot_data(model, discriminator, fold, syst_name=syst_name, regex=regex)
    else: return make_all_plot_data(model, discriminator, syst_name=syst_name, regex=regex)
def make_all_plot_data(model: Model, discriminator: str, syst_name: str = 'nominal', regex: str | list[str] = ''):
    weights = model.dfdataset.get_all_test(syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}eventWeight"].to_numpy()
    labels = model.dfdataset.get_all_test(syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}label1D"].to_numpy()
    samples = model.dfdataset.get_all_test(syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}sample_name"].to_numpy()

    predictions = model.evaluate_all_folds(syst_name=syst_name, regex=regex)
    transformed_labels, transformed_predictions, _ = transform_preds(model.dfdataset.class_sample_map.keys(), discriminator, predictions)

    return transformed_labels, {'preds': transformed_predictions, 'weights': weights, 'labels': labels, 'samples': samples}
def make_fold_plot_data(model: Model, discriminator: str, fold: int, syst_name: str = 'nominal', regex: str | list[str] = ''):
    weights = model.dfdataset.get_test(fold, syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}eventWeight"].to_numpy()
    labels = model.dfdataset.get_test(fold, syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}label1D"].to_numpy()
    samples = model.dfdataset.get_test(fold, syst_name=syst_name, regex=regex).loc[:, f"{model.dfdataset.aux_var_prefix}sample_name"].to_numpy()

    predictions = model.evaluate(fold, syst_name=syst_name, regex=regex)
    transformed_labels, transformed_predictions, _ = transform_preds(model.dfdataset.class_sample_map.keys(), discriminator, predictions)

    return transformed_labels, {'preds': transformed_predictions, 'weights': weights, 'labels': labels, 'samples': samples}

def split_plot_data_by_class(model: Model, plot_data: dict):
    class_plot_data = {}
    for j, class_name in enumerate(model.dfdataset.class_sample_map.keys()):
        event_mask = (plot_data['labels'] == j)
        if np.all(~event_mask): continue
        class_plot_data[class_name] = {
            key: value[event_mask] for key, value in plot_data.items()
        }
    return class_plot_data
def split_plot_data_by_sample(model: Model, plot_data: dict):
    sample_plot_data = {}
    for sample_name in np.unique(plot_data['samples']):
        event_mask = (plot_data['sample_name'] == sample_name)
        if np.all(~event_mask): continue
        sample_plot_data[sample_name] = {
            key: value[event_mask] for key, value in plot_data.items()
        }
    return sample_plot_data
