# Stdlib packages
import os

# Common Py packages
import numpy as np
import pandas as pd

# ML packages
from sklearn.model_selection import train_test_split

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import match_sample, match_regex

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
            mask = df[col].gt(0)
            df.loc[mask, col] = np.log(df.loc[mask, col])


def compute_zscore(masked_x: np.ma.MaskedArray):
    x_mean = masked_x.mean(axis=0)
    x_std = masked_x.std(axis=0)
    return x_mean.tolist(), x_std.tolist()

def apply_zscore(masked_x: np.ma.MaskedArray, stddict: dict):
    for i, col in enumerate(stddict['col']):
        masked_x[:, i] = (masked_x[:, i] - stddict['mean'][i]) / stddict['std'][i]


#############################################################
# Process train/test split
def equalProc_train_test_split(df: pd.DataFrame, train_size: float|None=None, test_size: float|None=None, random_state: int|None=None, shuffle: bool=True, stratify: object|None=None):
    sample_name_col = match_regex('sample_name', df.columns)
    unique_procs = pd.unique(df[sample_name_col])

    train_df, val_df = pd.DataFrame(columns=df.columns).astype(df.dtypes), pd.DataFrame(columns=df.columns).astype(df.dtypes)
    for proc in unique_procs:
        train_proc_df, val_proc_df = train_test_split(
            df.loc[df[sample_name_col].eq(proc)], 
            train_size=train_size, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify
        )
        train_df = pd.concat([train_df, train_proc_df], ignore_index=True); val_df = pd.concat([val_df, val_proc_df], ignore_index=True)
    return train_df, val_df


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
def make_output_filepath(filepath: str, base_output_dirpath: str, extra_text: str):
    filename = filepath[filepath.rfind('/')+1:]
    output_dirpath = os.path.join(base_output_dirpath, filepath[:filepath.rfind('/')])

    filename = filename[:filename.rfind('.')] + f"_{extra_text}" + filename[filename.rfind('.'):]

    return os.path.join(output_dirpath, filename)


