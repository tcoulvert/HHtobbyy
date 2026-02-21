# Stdlib packages
import copy
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

NONRES_RESCALE = {
    '2016*preVFP': {
        'DDQCDGJets': 1.1233,
        'GGJets': 1.1303,
    },
    '2016*postVFP': {
        'DDQCDGJets': 1.2077,
        'GGJets': 1.3069,
    },
    '2017': {
        'DDQCDGJets': 1.1817, 
        'GGJets': 1.2002,
    },
    '2018': {
        'DDQCDGJets': 1.1495, 
        'GGJets': 1.2153,
    },
    '2022*preEE': {
        'DDQCDGJets': 1.0498,
        'GGJets': 1.3980,
    },
    '2022*postEE': {
        'DDQCDGJets': 1.0896,
        'GGJets': 1.3762,
    },
    '2023*preBPix': {
        'DDQCDGJets': 1.0369,
        'GGJets': 1.4987,
    },
    '2023*postBPix': {
        'DDQCDGJets': 1.1814,
        'GGJets': 1.5032,
    },
    '2024': {
        'DDQCDGJets': 1.2140,
        'GGJets': 1.5925,
    }
}
for era, rescale_dict in NONRES_RESCALE.items(): rescale_dict['!DDQCDGJet*GJet'] = 2.6


################################


def format_class_names(class_names):
    return [class_name.replace(' ', '').replace('+', '') for class_name in class_names]

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

def argsorted(objects):
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
def get_train_Dataframe(dataset_dirpath: str, fold_idx: int, dataset: str="train", minimal: bool=True):
    df_list = []
    aux_list = []

    filepaths = get_train_filepaths_func(dataset_dirpath, dataset=dataset)(fold_idx)
    if minimal: filepaths = {bdt_class: [filepaths[bdt_class][0]] for bdt_class in filepaths.keys()}

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

    assert 'Data' not in set(np.unique(aux['AUX_sample_name']).tolist()), f"Data is getting into train dataset... THIS IS VERY BAD"

    # Upweight resonant background and signal samples for training #
    # Non-Resonant background #
    for era, nonres_rescale_dict in NONRES_RESCALE.items():
        for sample_name, sample_rescale in nonres_rescale_dict.items():
            era_sample_mask = np.ones_like(aux['AUX_mass'])
            for sub_era in era.split('*'):
                if len(sub_era) == 0: continue
                elif sub_era[0] == '!':
                    era_sample_mask = np.logical_and(era_sample_mask, ~aux['AUX_sample_era'].str.contains(sub_era, regex=True).to_numpy())
                else:
                    era_sample_mask = np.logical_and(era_sample_mask, aux['AUX_sample_era'].str.contains(sub_era, regex=True).to_numpy())
            for sub_sample_name in sample_name.split('*'):
                if len(sub_sample_name) == 0: continue
                elif sub_sample_name[0] == '!':
                    era_sample_mask = np.logical_and(era_sample_mask, ~aux['AUX_sample_name'].str.contains(sub_sample_name, regex=True).to_numpy())
                else:
                    era_sample_mask = np.logical_and(era_sample_mask, aux['AUX_sample_name'].str.contains(sub_sample_name, regex=True).to_numpy())
            aux.loc[era_sample_mask, 'AUX_eventWeight'] = aux.loc[era_sample_mask, 'AUX_eventWeight'] * sample_rescale
            aux.loc[era_sample_mask, 'AUX_eventWeightTrain'] = aux.loc[era_sample_mask, 'AUX_eventWeight']

    # Resonant background
    res_class_mask = aux['AUX_label1D'].eq(-1).to_numpy()
    for i, key in enumerate(filepaths.keys()):
        if match_sample(key, ['ttH', 'VH']) is not None:
            res_class_mask = np.logical_or(res_class_mask, aux['AUX_label1D'].eq(i).to_numpy())
    aux.loc[res_class_mask, 'AUX_eventWeightTrain'] = aux.loc[res_class_mask, 'AUX_eventWeightTrain'] * RES_BKG_RESCALE

    # Signal
    signal_class_mask = aux['AUX_label1D'].eq(-1).to_numpy()
    for i, key in enumerate(filepaths.keys()):
        if match_sample(key, ['HH']) is not None:
            signal_class_mask = np.logical_or(signal_class_mask, aux['AUX_label1D'].eq(i).to_numpy())
    aux.loc[signal_mask, 'AUX_eventWeightTrain'] = aux.loc[signal_class_mask, 'AUX_eventWeightTrain'] * np.sum(aux.loc[~signal_class_mask, 'AUX_eventWeightTrain']) / np.sum(aux.loc[signal_class_mask, 'AUX_eventWeightTrain'])
    
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
def get_train_DMatrices(dataset_dirpath: str, fold_idx: int, val_split: float=0.2, dataset: str='all', **kwargs):
    if 'res_bkg_rescale' in kwargs: RES_BKG_RESCALE = kwargs['res_bkg_rescale']
    if 'shuffle' in kwargs: DF_SHUFFLE = kwargs['shuffle']
    
    if dataset in ['all', 'train', 'val']:
        tr_df, tr_aux = get_train_Dataframe(dataset_dirpath, fold_idx)
        train_df, val_df, train_aux, val_aux = train_test_split(tr_df, tr_aux, test_size=val_split, random_state=RNG_SEED)
        train_dm = get_DMatrix(train_df, train_aux)
        val_dm = get_DMatrix(val_df, val_aux)
    if dataset in ['all', 'test']:
        test_df, test_aux = get_train_Dataframe(dataset_dirpath, fold_idx, 'test')
        test_dm = get_DMatrix(test_df, test_aux, dataset='test')

    if dataset == 'all': return train_dm, val_dm, test_dm
    elif dataset == 'train': return train_dm
    elif dataset == 'val': return val_dm
    elif dataset == 'test': return test_dm
    else: raise ValueError(f"Unknown dataset: {dataset}")

def get_test_DMatrix(filepath: str):
    df, aux = get_Dataframes(filepath)
    return get_DMatrix(df, aux, dataset='test', label=False)

def get_test_subset_Dataframes(dataset_dirpath: str, fold_idx: int, regexs: list, label: bool=True, minimal: bool=False):
    class_sample_map = get_class_sample_map(dataset_dirpath)

    df_full, aux_full = None, None
    for regex in regexs:
        
        filepaths = [
            filepath for filepath in get_test_filepaths_func(dataset_dirpath)(fold_idx)['test']
            if match_sample(filepath[len(dataset_dirpath):], [regex]) is not None
        ]
        if minimal: filepaths = [filepaths[0]]
        df, aux = None, None
        for filepath in filepaths:
            if df is None: 
                df, aux = get_Dataframes(filepath)
            else: 
                new_df, new_aux = get_Dataframes(filepath)
                df, aux = pd.concat([df, new_df]), pd.concat([aux, new_aux])
                
        if match_sample(regex, ['Data']) is not None:
            SR_mask = np.logical_and(aux.loc[:, 'AUX_mass'].to_numpy() > 120., aux.loc[:, 'AUX_mass'].to_numpy() < 130.)
            df, aux = df.loc[~SR_mask], aux.loc[~SR_mask]
            if label:
                data_idx = [i for i, key in enumerate(class_sample_map.keys()) if 'nonres' in key.lower()][0]
                aux['AUX_label1D'] = data_idx * np.ones_like(aux['AUX_mass'])
        elif match_sample(regex, ['SherpaNLO']) is not None and label:
            sherpa_idx = [i for i, key in enumerate(class_sample_map.keys()) if 'nonres' in key.lower()][0]
            aux['AUX_label1D'] = sherpa_idx * np.ones_like(aux['AUX_mass'])
        elif label:
            class_idx = [i for i, regex_list in enumerate(class_sample_map.values()) if regex in regex_list][0]
            aux['AUX_label1D'] = class_idx * np.ones_like(aux['AUX_mass'])

        if df_full is None:
            df_full, aux_full = copy.deepcopy(df), copy.deepcopy(aux)
        else:
            df_full, aux_full = pd.concat([df_full, df]), pd.concat([aux_full, aux])

    return df_full, aux_full

def get_test_subset_DMatrix(dataset_dirpath: str, fold_idx: int, regexs: list, label: bool=True):
    df_full, aux_full = get_test_subset_Dataframes(dataset_dirpath, fold_idx, regexs, label=label)
    return get_DMatrix(df_full, aux_full, dataset='test', label=label)