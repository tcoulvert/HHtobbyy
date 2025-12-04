# Stdlib packages
import glob
import json
import os
import re

# Common Py packages
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# HEP packages
import xgboost as xgb

# ML packages
from sklearn.model_selection import train_test_split

################################

from preprocessing_utils import match_sample

################################


RES_BKG_RESCALE = 100
DF_SHUFFLE = True
RNG_SEED = 21
FILL_VALUE = -999

using_resolution_var = False
do_reweight = False
reweight_var = 'sigma_m_over_m_smeared_decorr'
################################


def format_class_names(class_names):
    return [''.join(class_name.split(' ')) for class_name in class_names]

def get_class_sample_map(dataset_dirpath: str):
    class_sample_map_filepath = os.path.join(dataset_dirpath, "class_sample_map.json")
    with open(class_sample_map_filepath, "r") as f:
        class_sample_map = json.load(f)
    return class_sample_map

def get_n_folds(dataset_dirpath: str):
    filepaths = glob.glob(os.path.join(dataset_dirpath, "**", f"*train*.parquet"), recursive=True)
    print(filepaths)
    max_fold = max([
        int(filepath[re.search('train[0-9]', filepath).end()-1]) for filepath in filepaths
    ])  # only works for up to 10 folds -- currently using 5, not likely to increase due to low-stats
    return max_fold + 1
    

def get_train_filepaths_func(dataset_dirpath: str, dataset: str="train", syst_name: str='nominal'):
    class_sample_map = get_class_sample_map(dataset_dirpath)
    return lambda fold_idx: {
        class_name: sorted(
            set(
                [
                    sample_filepath 
                    for sample_filepath in glob.glob(os.path.join(dataset_dirpath, "**", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
                    if match_sample(sample_filepath, sample_names) is not None
                ]
            )
        ) for class_name, sample_names in class_sample_map.items()
    }
def get_test_filepaths_func(dataset_dirpath: str, syst_name: str='nominal'):
    return lambda fold_idx: {
        'test': sorted(
            set(
                glob.glob(os.path.join(dataset_dirpath, "**", f"*{syst_name}*", f"*test{fold_idx}*.parquet"), recursive=True)
            ) | (
                set(glob.glob(os.path.join(dataset_dirpath, "**", f"Data*test{fold_idx}*.parquet"), recursive=True))
                if syst_name == "nominal" else set()
            )
        )
    }

def argsorted(objects, **kwargs):
    object_to_index = {}
    for index, object in enumerate(objects):
        object_to_index[object] = index
    sorted_objects = sorted(objects)
    sorted_indices = [object_to_index[object] for object in sorted_objects]
    return sorted_indices

def get_labelND(label1D):
    """
    Returns the ND label vector (one-hot encoded) from the 1D label vector (integer encoded).
    """
    return np.tile(np.arange(np.max(label1D)), (np.size(label1D), 1)) == label1D
def get_label1D(labelND):
    """
    Returns the 1D label vector (integer encoded) from the ND label vector (one-hot encoded).
    """
    return np.nonzero(labelND).flatten()

def get_Dataframe(filepath: str, aux: bool=False):
    schema = pq.read_schema(filepath)
    vars = [var for var in schema.names if ('AUX_' not in var) ^ aux]

    if aux and do_reweight:
        vars += [reweight_var]
        
    df = pq.read_table(filepath, columns=vars).to_pandas()

    return df
def get_Dataframes(filepath: str):
    return get_Dataframe(filepath), get_Dataframe(filepath, aux=True)
def get_train_Dataframe(dataset_dirpath: str, fold_idx: int, dataset: str="train"):
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_filepath = caller_frame.f_code.co_filename
    caller_filename = os.path.basename(caller_filepath)
    filepaths = get_train_filepaths_func(dataset_dirpath, dataset=dataset)(fold_idx)

    df, aux = None, None
    for i, bdt_class in enumerate(filepaths.keys()):
        class_df = pd.concat(
            [get_Dataframe(filepath) for filepath in filepaths[bdt_class]], ignore_index=True
        )
        class_aux = pd.concat(
            [get_Dataframe(filepath, aux=True) for filepath in filepaths[bdt_class]], ignore_index=True
        )
        class_aux['AUX_label1D'] = i
        class_aux['AUX_eventWeightTrain'] = class_aux['AUX_eventWeight']

        mask_field = [field for field in class_aux.columns if 'resolved_BDT_mask' in field][0]
        mask = (class_aux[mask_field] > 0)
        class_df = class_df.loc[mask]
        class_aux = class_aux.loc[mask]
        
        if df is None:
            df = class_df
            aux = class_aux
        else:
            df = pd.concat([df, class_df], ignore_index=True)
            aux = pd.concat([aux, class_aux], ignore_index=True)
    
    plot_mode = False
    if 'plot' in caller_filename.lower():
        plot_mode = True
    
    if do_reweight and not plot_mode:
        # from AUX_sample_name column get all of the available sample names
        sample_names = aux['AUX_sample_name'].unique().tolist()
        for sample_name in sample_names:
            sample_mask = aux['AUX_sample_name'].eq(sample_name)
            sum_of_weights = aux.loc[sample_mask, 'AUX_eventWeightTrain'].sum()
            # import deepcopy
            import copy
            reweighted_events = copy.deepcopy(aux.loc[sample_mask, reweight_var])
            # set the FILL_VALUE entries to 1 to avoid division by zero
            # and this value is standardized before training so we need to invert the standardization first
            invalid_mask = reweighted_events < (FILL_VALUE+2)
            var_mean = 0.014581804722822288
            var_std = 0.0064496533424058715
            original_values = reweighted_events * var_std + var_mean
            original_values[invalid_mask] = 1.0  # set invalid entries to 1.0
            original_values[original_values <= 0] = 1.0  # avoid division by zero
            aux.loc[sample_mask, 'AUX_eventWeightTrain'] = aux.loc[sample_mask, 'AUX_eventWeightTrain'] / original_values
            new_sum_of_weights = aux.loc[sample_mask, 'AUX_eventWeightTrain'].sum()
            _reweight_factor = sum_of_weights / new_sum_of_weights
            aux.loc[sample_mask, 'AUX_eventWeightTrain'] = aux.loc[sample_mask, 'AUX_eventWeightTrain'] * _reweight_factor
            print(f"Applied reweighting using {reweight_var} for sample {sample_name} with factor {_reweight_factor} (just for check)")
        # remove reweight_var column from df
        df = df.drop(columns=[reweight_var])

    if not using_resolution_var:
        # remove resolution variable from df
        if reweight_var in df.columns:
            df = df.drop(columns=[reweight_var])

    if reweight_var in aux.columns:
        aux = aux.drop(columns=[reweight_var])


    # Upweight resonant background and signal samples for training #
    # Non-Resonant background #
    nonres_mask = aux['AUX_sample_name'].eq('GJet')
    aux.loc[nonres_mask, 'AUX_eventWeightTrain'] = aux.loc[nonres_mask, 'AUX_eventWeightTrain'] * 2.78  # 1.78
    # Resonant background
    for i, _ in enumerate([key for key in filepaths.keys() if 'nonRes' not in key and 'HH' not in key]):
        class_mask = aux['AUX_label1D'].eq(i)
        aux.loc[class_mask, 'AUX_eventWeightTrain'] = aux.loc[class_mask, 'AUX_eventWeightTrain'] * RES_BKG_RESCALE
    # Signal
    for i, _ in enumerate([key for key in filepaths.keys() if 'HH' in key]):
        signal_mask = aux['AUX_label1D'].eq(i)
        background_mask = aux['AUX_label1D'].ne(i)
        aux.loc[signal_mask, 'AUX_eventWeightTrain'] = aux.loc[signal_mask, 'AUX_eventWeightTrain'] * np.sum(aux.loc[background_mask, 'AUX_eventWeightTrain']) / np.sum(aux.loc[signal_mask, 'AUX_eventWeightTrain'])
    
    if DF_SHUFFLE:
        rng = np.random.default_rng(seed=RNG_SEED)
        class_shuffle_idx = rng.permutation(df.index)
        df.reindex(class_shuffle_idx)
        aux.reindex(class_shuffle_idx)

    return df, aux
def get_test_Dataframe(filepath: str):
    return pq.read_table(filepath).to_pandas()

def get_DMatrix(df, aux, dataset: str='train', label: bool=True):
    if label: label_arg = aux['AUX_label1D']
    else: label_arg = None
    return xgb.DMatrix(
        data=df, label=label_arg, weight=np.abs(aux['AUX_eventWeightTrain'] if dataset.lower() == 'train' else aux['AUX_eventWeight']),
        missing=FILL_VALUE, feature_names=list(df.columns)
    )
def get_train_DMatrices(dataset_dirpath: str, fold_idx: int, val_split: float=0.2, **kwargs):
    if 'res_bkg_rescale' in kwargs: RES_BKG_RESCALE = kwargs['res_bkg_rescale']
    if 'shuffle' in kwargs: DF_SHUFFLE = kwargs['shuffle']

    tr_df, tr_aux = get_train_Dataframe(dataset_dirpath, fold_idx)
    train_df, val_df, train_aux, val_aux = train_test_split(tr_df, tr_aux, test_size=val_split, random_state=RNG_SEED)
    test_df, test_aux = get_train_Dataframe(dataset_dirpath, fold_idx, 'test')

    train_dm = get_DMatrix(train_df, train_aux)
    val_dm = get_DMatrix(val_df, val_aux)
    test_dm = get_DMatrix(test_df, test_aux, dataset='test')

    return train_dm, val_dm, test_dm
def get_test_DMatrix(filepath: str):
    df, aux = get_Dataframes(filepath)
    return get_DMatrix(df, aux, dataset='test', label=False)
