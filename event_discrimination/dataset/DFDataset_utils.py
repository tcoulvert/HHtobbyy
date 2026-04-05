# Stdlib packages
import glob
import os

# Common Py packages
import numpy as np
import pandas as pd

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import (
    match_sample, match_regex
)


#############################################################
# Standardization
def no_standardize(column: str):
    no_std_terms = {
        'phi', 'eta',  # angular
        'id',  # IDs
        'btag'  # bTags
    }
    return any(no_std_term in column.lower() for no_std_term in no_std_terms)

def log_standardize(column: str):
    log_std_terms = {
        'pt', 'chi',
    }
    return any(log_std_term in column for log_std_term in log_std_terms)

def apply_logs(df: pd.DataFrame):
    for col in df.columns:
        if log_standardize(col):
            mask = (df[col].to_numpy() > 0)
            df.loc[mask, col] = np.log(df.loc[mask, col])
    return df


#############################################################
# Label mapping
def get_labelND(label1D: np.ndarray):
    """
    Returns the ND label vector (one-hot encoded) from the 1D label vector (integer encoded).
    """
    return np.tile(np.arange(np.max(label1D)), (np.size(label1D), 1)) == np.stack([label1D for _ in np.arange(np.max(label1D))]).T
def get_label1D(labelND: np.ndarray):
    """
    Returns the 1D label vector (integer encoded) from the ND label vector (one-hot encoded).
    """
    return np.nonzero(labelND).flatten()


#############################################################
# Class-file mapping
def map_filepath_to_class(class_sample_map: dict, filepath: str):
    mapped_class_idx = -1
    for i, (class_name, sample_names) in enumerate(class_sample_map.items()):
        if match_sample(filepath, sample_names) is not None: 
            mapped_class_idx = i; break
    return mapped_class_idx


#############################################################
# Output train/test files
def make_output_filepath(filepath, base_output_dirpath, extra_text):
    filename = filepath[filepath.rfind('/')+1:]
    output_dirpath = os.path.join(base_output_dirpath, filepath[:filepath.rfind('/')])
    os.makedirs(output_dirpath, exist_ok=True)

    filename = filename[:filename.rfind('.')] + f"_{extra_text}" + filename[filename.rfind('.'):]

    return os.path.join(output_dirpath, filename)


#############################################################
def check_train_filepaths(train_filepaths: list, eras: list):
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

def get_input_filepaths(eras: str|list[str], regex: str|list[str]=""):
    if type(eras) is str: eras = get_era_filepaths(eras)
    input_filepaths = []
    
    for era in eras:
        sample_filepaths = glob.glob(os.path.join(era, "**", f"*{END_FILEPATH}"), recursive=True)
        for sample_filepath in sample_filepaths:
            sub_sample_filepath = sample_filepath[len(era):]
            if match_sample(
                sub_sample_filepath, 
                {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}
            ) is not None:
                input_filepaths.append(sample_filepath)
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