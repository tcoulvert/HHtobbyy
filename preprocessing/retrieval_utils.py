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

from preprocessing_utils import match_sample, match_regex

################################


RES_BKG_RESCALE = 100
DF_SHUFFLE = True
RNG_SEED = 21
FILL_VALUE = -999

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
    max_fold = max([
        int(filepath[re.search('train[0-9]', filepath).end()-1]) for filepath in filepaths
    ])  # only works for up to 10 folds -- currently using 5, not likely to increase due to low-stats
    return max_fold + 1
    

def get_train_filepaths_func(dataset_dirpath: str, dataset: str="train", syst_name: str='nominal'):
    class_sample_map = get_class_sample_map(dataset_dirpath)
    return lambda fold_idx: {
        class_name: sorted(
            set(
                sample_filepath
                for sample_filepath in glob.glob(os.path.join(dataset_dirpath, "**", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
                if (
                    (syst_name == "nominal" and match_sample(sample_filepath[len(dataset_dirpath):], ["_up", "_down"]) is None) 
                    or match_sample(sample_filepath[len(dataset_dirpath):], [syst_name]) is not None
                ) and match_sample(sample_filepath[len(dataset_dirpath):], sample_names) is not None
            )
        ) for class_name, sample_names in class_sample_map.items()
    }
def get_test_filepaths_func(dataset_dirpath: str, syst_name: str='nominal'):
    return lambda fold_idx: {
        'test': sorted(
            set(
                sample_filepath
                for sample_filepath in glob.glob(os.path.join(dataset_dirpath, "**", f"*test{fold_idx}*.parquet"), recursive=True)
                if ( 
                    (syst_name == "nominal" and match_sample(sample_filepath[len(dataset_dirpath):], ["_up", "_down"]) is None) 
                    or match_sample(sample_filepath[len(dataset_dirpath):], [syst_name]) is not None
                )
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

def get_Dataframe(filepath: str, aux: bool=False, n_folds_fold_idx: tuple=None, columns: list=None):
    schema = pq.read_schema(filepath)
    if columns is None:
        columns = [var for var in schema.names if ('AUX_' not in var) ^ aux]
    else:
        assert set(columns) <= set(schema.names), f"The requested list of columns are not all in the file. Missing variables:\n{set(columns) - set(schema.names)}"

    df = pq.read_table(filepath, columns=columns).to_pandas()

    if n_folds_fold_idx is None:
        return df
    else:
        AUX_event_df = pq.read_table(filepath, columns=['AUX_event']).to_pandas()
        mask = (AUX_event_df.loc[:, 'AUX_event'] % n_folds_fold_idx[0]).eq(n_folds_fold_idx[1])
        return df.loc[mask].reset_index(drop=True)
def get_Dataframes(filepath: str, n_folds_fold_idx: tuple=None):
    return get_Dataframe(filepath, n_folds_fold_idx=n_folds_fold_idx), get_Dataframe(filepath, aux=True, n_folds_fold_idx=n_folds_fold_idx)
def get_train_Dataframe(dataset_dirpath: str, fold_idx: int, dataset: str="train", **kwargs):
    df_list = []
    aux_list = []

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
        
        df_list.append(class_df)
        aux_list.append(class_aux)
    
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        aux = pd.concat(aux_list, ignore_index=True)
    else:
        return None, None

    if kwargs.get('restandardize', False):
        if 'previous_std' in kwargs and 'new_std' in kwargs:
            print(f"[INFO] Re-standardizing variables from {kwargs['previous_std']} to {kwargs['new_std']}")
            df = reStandardize_variable(df, kwargs['previous_std'], kwargs['new_std'])

    assert 'Data' not in set(np.unique(aux['AUX_sample_name']).tolist()), f"Data is getting into train dataset... THIS IS VERY BAD"

    # Upweight resonant background and signal samples for training #
    # Non-Resonant background #
    if match_regex('DDQCDGJets', [filepath for filepath_class_list in filepaths.values() for filepath in filepath_class_list]) is not None:
        DDQCD_GGJET_2024_mask = aux['AUX_sample_name'].eq('GGJets')
        aux.loc[DDQCD_GGJET_2024_mask, 'AUX_eventWeight'] = aux.loc[DDQCD_GGJET_2024_mask, 'AUX_eventWeight'] * 1.59
        aux.loc[DDQCD_GGJET_2024_mask, 'AUX_eventWeightTrain'] = aux.loc[DDQCD_GGJET_2024_mask, 'AUX_eventWeight']
    # Resonant background
    for i, key in enumerate(filepaths.keys()):
        if 'ttH' in key or 'VH' in key:
            class_mask = aux['AUX_label1D'].eq(i)
            aux.loc[class_mask, 'AUX_eventWeightTrain'] = aux.loc[class_mask, 'AUX_eventWeightTrain'] * RES_BKG_RESCALE

    # Sherpa: scale it to same as nonRes background
    sherpa_mask = aux['AUX_sample_name'].eq('SherpaNLO')
    if any(sherpa_mask):
        # scale shepra by 1000, as mistake in cross-section fb vs pb
        aux.loc[sherpa_mask, 'AUX_eventWeight'] *= 1000
        aux.loc[sherpa_mask, 'AUX_eventWeightTrain'] *= 1000
        # nonres_label = [i for i, key in enumerate(filepaths.keys()) if 'nonRes' in key][0]
        # nonres_mask = aux['AUX_label1D'].eq(nonres_label)
        # nonsherpa_mask = aux['AUX_sample_name'].ne('SherpaNLO')
        # nonres_nonsherpa_mask = nonres_mask & nonsherpa_mask
        # sum_nonres = np.sum(aux.loc[nonres_nonsherpa_mask, 'AUX_eventWeightTrain'])
        # sum_sherpa = np.sum(aux.loc[sherpa_mask, 'AUX_eventWeightTrain'])
        # if sum_sherpa > 0:
        #     print(f"[INFO] Scaling SherpaNLO weights from {sum_sherpa} to {sum_nonres} for training")
        #     aux.loc[sherpa_mask, 'AUX_eventWeightTrain'] *= (sum_nonres / sum_sherpa)

    # Signal
    for i, key in enumerate(filepaths.keys()):
        if 'HH' in key:
            signal_mask = aux['AUX_label1D'].eq(i)
            background_mask = aux['AUX_label1D'].ne(i)
            aux.loc[signal_mask, 'AUX_eventWeightTrain'] = aux.loc[signal_mask, 'AUX_eventWeightTrain'] * np.sum(aux.loc[background_mask, 'AUX_eventWeightTrain']) / np.sum(aux.loc[signal_mask, 'AUX_eventWeightTrain'])
    
    # check average weight and sum weight for each class
    for i, class_name in enumerate(filepaths.keys()):
        class_mask = aux['AUX_label1D'].eq(i)
        sum_weights = aux.loc[class_mask, 'AUX_eventWeightTrain'].sum()
        avg_weight = aux.loc[class_mask, 'AUX_eventWeightTrain'].mean()
        print(f"[CHECK] Class {class_name} (label {i}): number of events = {class_mask.sum()}, sum of weights = {sum_weights}, average weight = {avg_weight}")

    if DF_SHUFFLE:
        rng = np.random.default_rng(seed=RNG_SEED)
        class_shuffle_idx = rng.permutation(df.index)
        df.reindex(class_shuffle_idx)
        aux.reindex(class_shuffle_idx)

    return df, aux
def get_test_Dataframe(filepath: str):
    df, aux = get_Dataframes(filepath)
    for col in aux.columns: df[col] = aux[col]
    return df

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

    if kwargs.get('test_only', False):
        # save resource when only need test DMatrix
        test_df, test_aux = get_train_Dataframe(dataset_dirpath, fold_idx, 'test', **kwargs)
        test_dm = get_DMatrix(test_df, test_aux, dataset='test')
        if kwargs.get('get_aux', False):
            return None, None, test_dm, None, None, test_aux
        return None, None, test_dm
    tr_df, tr_aux = get_train_Dataframe(dataset_dirpath, fold_idx, **kwargs)
    train_df, val_df, train_aux, val_aux = train_test_split(tr_df, tr_aux, test_size=val_split, random_state=RNG_SEED)
    test_df, test_aux = get_train_Dataframe(dataset_dirpath, fold_idx, 'test', **kwargs)

    train_dm = get_DMatrix(train_df, train_aux)
    val_dm = get_DMatrix(val_df, val_aux)
    test_dm = get_DMatrix(test_df, test_aux, dataset='test')

    if not kwargs.get('get_aux', False):
        return train_dm, val_dm, test_dm
    return train_dm, val_dm, test_dm, train_aux, val_aux, test_aux
def get_test_DMatrix(filepath: str):
    df, aux = get_Dataframes(filepath)
    return get_DMatrix(df, aux, dataset='test', label=False)

def reStandardize_variable(df, previous_std_json, new_std_json, fill_value: float=FILL_VALUE):
    def load_std_dict(std_input):
        if isinstance(std_input, str):
            with open(std_input, 'r') as f:
                data = json.load(f)
        else:
            data = std_input
            
        if isinstance(data, dict) and "col" in data and "mean" in data and "std" in data:
            return {
                c: {"mean": m, "std": s} 
                for c, m, s in zip(data["col"], data["mean"], data["std"])
            }
        return data

    previous_std = load_std_dict(previous_std_json)
    new_std = load_std_dict(new_std_json)

    for col in df.columns:
        if col in previous_std and col in new_std:
            # valid values are those that are not the fill value
            # assuming fill_value is a large negative number like -999, consistent with other functions
            valid_mask = df[col] > (fill_value + 2)

            prev_mean = previous_std[col]['mean']
            prev_std_val = previous_std[col]['std']
            new_mean = new_std[col]['mean']
            new_std_val = new_std[col]['std']

            # de-standardize: x_raw = x_std * std + mean
            raw_values = df.loc[valid_mask, col] * prev_std_val + prev_mean
            # re-standardize: x_new_std = (x_raw - new_mean) / new_std
            df.loc[valid_mask, col] = (raw_values - new_mean) / new_std_val
        elif col not in new_std:
            print(f"[WARNING] Column {col} not found in new_std dictionary. Dropping this column.")
            df.drop(columns=[col], inplace=True)
    return df