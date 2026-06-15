# Stdlib packages
import os

# Common Py packages
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ML packages
from sklearn.model_selection import train_test_split

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import match_sample, match_regex, sub_filepath

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

def apply_zscore(masked_x: np.ma.MaskedArray, mean: float, stddev: float):
    masked_x = (masked_x - mean) / stddev


#############################################################
# Process train/test split
def random_oversample(df: pd.DataFrame, rng_seed: int=21):
    sample_name_col = match_regex('sample_name', df.columns)
    unique_procs = pd.unique(df[sample_name_col])
    largest_proc = np.max([np.sum(df[sample_name_col].eq(proc)) for proc in unique_procs])
    
    for proc in unique_procs:
        proc_idxs = df[df[sample_name_col].eq(proc)].index
        if len(proc_idxs) == largest_proc: continue

        rand_idxs = np.random.default_rng(seed=rng_seed).choice(proc_idxs, size=largest_proc-len(proc_idxs))
        df.merge(df.iloc[rand_idxs])

def random_undersample(df: pd.DataFrame, rng_seed: int=21):
    sample_name_col = match_regex('sample_name', df.columns)
    unique_procs = pd.unique(df[sample_name_col])
    smallest_proc = np.min([np.sum(df[sample_name_col].eq(proc)) for proc in unique_procs])
    
    for proc in unique_procs:
        proc_idxs = df[df[sample_name_col].eq(proc)].index
        if len(proc_idxs) == smallest_proc: continue

        rand_idxs = np.random.default_rng(seed=rng_seed).choice(proc_idxs, size=len(proc_idxs)-smallest_proc)
        df.drop(rand_idxs)


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


#############################################################
# Decorators for batch handling
def batched_writer(func):
    def wrapper(pq_iter_func, infilepath: str, outfilepath: str, *args, **kwargs):
        eos_infilepath = eos.load_file_eos(infilepath, **kwargs)
        eos_outfilepath = eos.save_file_eos(outfilepath, **kwargs)
        pq_writer = None
        for pq_batch in pq_iter_func(eos_infilepath, **kwargs):
            table_batch = pa.Table.from_pandas(func(pq_batch.to_pandas(), *args, **kwargs))
            if pq_writer is None: pq_writer = pq.ParquetWriter(eos_outfilepath, schema=table_batch.schema)
            pq_writer.write_table(table_batch)
        if pq_writer is not None: pq_writer.close()
        eos.delete_lockfile(eos_infilepath); eos.delete_lockfile(eos_outfilepath)
    return wrapper

def batched_loader(func):
    def wrapper(pq_iter_func, filepath: str, *args, **kwargs):
        eos_filepath = eos.load_file_eos(filepath, **kwargs)
        df_batches = []
        for pq_batch in pq_iter_func(eos_filepath, **kwargs):
            df_batches.append(func(pq_batch.to_pandas(), *args, **kwargs))
        eos.delete_lockfile(eos_filepath)
        return pd.concat(df_batches, ignore_index=True)
    return wrapper

def batched_executor(func):
    def wrapper(pq_iter_func, filepath: str, *args, **kwargs):
        eos_filepath = eos.load_file_eos(filepath, **kwargs)
        for pq_batch in pq_iter_func(eos_filepath, **kwargs):
            func(pq_batch.to_pandas(), *args, **kwargs)
        eos.delete_lockfile(eos_filepath)
    return wrapper