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





#############################################################
def get_era_filepaths(input_eras: str, split_data_mc_eras: bool=False):
    MC_eras, Data_eras = set(), set()
    with open(input_eras, 'r') as f:
        for line in f:
            stdline = line.strip()
            if len(stdline) == 0 or stdline[0] == '#': continue

            if 'sim' in stdline.lower(): MC_eras.add(stdline)
            elif 'data' in stdline.lower(): Data_eras.add(stdline)
            else: raise KeyError(f"Era {stdline} does not seem to be MC or Data, check the filepath is correct")
    if split_data_mc_eras: return MC_eras, Data_eras
    else: return (MC_eras | Data_eras)




#############################################################
def check_train_dataset(train_filepaths: list):
    good_dataset_bool = True
    for glob_name in [glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names]:
        for era in get_era_filepaths(args.input_eras, split_data_mc_eras=True)[0]:
            if 'data' in era: continue
            if match_regex(f"{era}*{glob_name}", train_filepaths) is None:
                if not (
                    ('2024' in era and glob_name in ['VH'])
                    or (('2022' in era or '2023' in era) and glob_name in ['ZH', 'Wm*H', 'Wp*H'])
                ):
                    good_dataset_bool = False; break
            elif match_regex(f"{era}*{glob_name}", list(set(train_filepaths) - set([match_regex(f"{era}*{glob_name}", train_filepaths)]))) is not None:
                if not 'GJet' in glob_name:
                    good_dataset_bool = False; break
        if not good_dataset_bool: break
    return good_dataset_bool

def get_input_filepaths():
    input_filepaths = {'train-test': list(), 'train': list(), 'test': list()}
    
    for era in ERAS:
        sample_filepaths = glob.glob(os.path.join(era, "**", f"*{END_FILEPATH}"), recursive=True)
        for sample_filepath in sample_filepaths:
            sub_sample_filepath = sample_filepath[len(era):]
            if (
                match_sample(sub_sample_filepath, TEST_ONLY_SAMPLES) is not None
                and match_sample(sub_sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is None
            ):
                input_filepaths['test'].append(sample_filepath)
            elif (
                match_sample(sub_sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is not None 
                and match_sample(sub_sample_filepath, TRAIN_ONLY_SAMPLES) is not None
            ):
                input_filepaths['train'].append(sample_filepath)
            elif match_sample(sub_sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is not None:
                input_filepaths['train-test'].append(sample_filepath)
            else:
                if DEBUG:
                    logger.warning(f"{sample_filepath} \nSample not found in any dict (TRAIN_TEST_SAMPLES, TRAIN_ONLY_SAMPLES, TEST_ONLY_SAMPLES). Continuing with other samples.")
                continue

    if not args.dont_check_dataset:
        assert check_train_dataset(input_filepaths['train-test']+input_filepaths['train']), f"Train dataset is missing some samples for some eras."
    
    return input_filepaths




#############################################################
def match_sample(sample_str, regexes):
    for regex in sorted(regexes, key=len, reverse=True):
        regex_bools = []
        match_str = sample_str
        for exp in regex.split('*'):
            if len(exp) == 0: continue
            if (exp[0] != '!' and re.search(exp.lower(), match_str.lower()) is not None):
                regex_bools.append(True)
                match_str = match_str[re.search(exp.lower(), match_str.lower()).end():]
            elif (exp[0] == '!' and re.search(exp[1:].lower(), match_str.lower()) is None):
                regex_bools.append(True)
            else:
                regex_bools.append(False)
        if all(regex_bools):
            return regex
        
def match_regex(regex, sample_strs):
    for sample_str in sorted(sample_strs, key=len):
        regex_bools = []
        match_str = sample_str
        for exp in regex.split('*'):
            if len(exp) == 0: continue
            if (exp[0] != '!' and re.search(exp.lower(), match_str.lower()) is not None):
                regex_bools.append(True)
                match_str = match_str[re.search(exp.lower(), match_str.lower()).end():]
            elif (exp[0] == '!' and re.search(exp[1:].lower(), match_str.lower()) is None):
                regex_bools.append(True)
            else:
                regex_bools.append(False)
        if all(regex_bools):
            return sample_str
        

        

#############################################################
def format_class_names(class_names):
    return [class_name.replace(' ', '').replace('+', '') for class_name in class_names]

def get_class_sample_map(dataset_dirpath: str):
    class_sample_map_filepath = os.path.join(dataset_dirpath, "class_sample_map.json")
    with open(class_sample_map_filepath, "r") as f:
        class_sample_map = json.load(f)
    return class_sample_map


#############################################################
def get_n_folds(dataset_dirpath: str):
    filepaths = glob.glob(os.path.join(dataset_dirpath, "**", f"*train*.parquet"), recursive=True)
    max_fold = max([
        int(filepath[re.search('train[0-9]', filepath).end()-1]) for filepath in filepaths
    ])  # only works for up to 10 folds -- currently using 5, not likely to increase due to low-stats
    return max_fold + 1




#############################################################
def get_traintest_filepaths_func(dataset_dirpath: str, dataset: str="train", syst_name: str='nominal'):
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
def get_test_filepaths_func(dataset_dirpath: str, syst_name: str='nominal', regex: str=''):
    return lambda fold_idx: {
        'test': sorted(
            set(
                sample_filepath
                for sample_filepath in glob.glob(os.path.join(dataset_dirpath, "**", f"*test{fold_idx}*.parquet"), recursive=True)
                if ( 
                    (syst_name == "nominal" and match_sample(sample_filepath[len(dataset_dirpath):], ["_up", "_down"]) is None) 
                    or match_sample(sample_filepath[len(dataset_dirpath):], [syst_name]) is not None
                ) and match_sample(sample_filepath[len(dataset_dirpath):], [regex]) is not None
            )
        )
    }



#############################################################
def argsorted(objects):
    object_to_index = {}
    for index, object in enumerate(objects):
        object_to_index[object] = index
    sorted_objects = sorted(objects)
    sorted_indices = [object_to_index[object] for object in sorted_objects]
    return sorted_indices




#############################################################
def get_DMatrix(df, aux, dataset: str='train', label: bool=True):
    if label: label_arg = aux['AUX_label1D']
    else: label_arg = None
    return xgb.DMatrix(
        data=df, label=label_arg, weight=np.abs(aux['AUX_eventWeightTrain'] if dataset.lower() == 'train' else aux['AUX_eventWeight']),
        missing=FILL_VALUE, feature_names=list(df.columns)
    )
