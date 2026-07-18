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
def no_standardize(column: str, no_std_regexs: list):
    return any(nostd in column.lower() for nostd in no_std_regexs)
def log_standardize(column: str, log_std_regexs: list):
    return any(logstd in column for logstd in log_std_regexs)

def identity(masked_x: np.ma.MaskedArray, *args):
    return masked_x
def nostd(masked_x: np.ma.MaskedArray, *args):
    count = masked_x.count()
    if count % 2 == 0: 
        value = 1; exp0_expsq_1 = []
    else:
        value = np.sqrt(count / (count - 1)); exp0_expsq_1 = [0]
    exp0_expsq_1 = exp0_expsq_1 + [value]*(count // 2) + [-value]*(count // 2)
    return np.ma.array(exp0_expsq_1)
def logstd(masked_x: np.ma.MaskedArray, *args):
    return np.ma.log(masked_x)
def logzscore(masked_x: np.ma.MaskedArray, column: str, no_std_regexs: list, log_std_regexs: list):
    if no_standardize(column, no_std_regexs): return nostd(masked_x)
    elif log_standardize(column, log_std_regexs): return logstd(masked_x)
    else: return identity(masked_x)


#############################################################
# Process train/test split
def equalProc(df: pd.DataFrame, **kwargs):
    sample_name_col = match_regex('sample_name', df.columns)
    unique_procs = pd.unique(df[sample_name_col])

    train_df, val_df = pd.DataFrame(columns=df.columns).astype(df.dtypes), pd.DataFrame(columns=df.columns).astype(df.dtypes)
    for proc in unique_procs:
        train_proc_df, val_proc_df = train_test_split(
            df.loc[df[sample_name_col].eq(proc)], **kwargs
        )
        train_df = pd.concat([train_df, train_proc_df], ignore_index=True); val_df = pd.concat([val_df, val_proc_df], ignore_index=True)
    return train_df, val_df
def scikit(df: pd.DataFrame, **kwargs):
    return train_test_split(df, **kwargs)


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
