# Stdlib packages
import glob
import os
import re
from threading import Thread 

# Common Py packages
import pyarrow as pa
import pyarrow.parquet as pq

# HEP packages
import eos_utils as eos

################################


FILL_VALUE = -999



#############################################################
def json_serialize(value):
    """
    Code copied from https://stackoverflow.com/a/42923092
    """
    return getattr(value, "tolist", lambda: value)()

#############################################################
def multifold(func, common_args, n_folds, parallel: bool=False, condor: bool=False, **kwargs):
    threads = []
    for fold in range(n_folds):
        args = (fold, )+common_args
        if parallel: 
            thread = Thread(target=func, name=f"Fold {fold}", args=args, kwargs=kwargs)
            thread.start(); threads.append(thread)
        elif condor:
            raise NotImplementedError(f"Multifold via Condor not yet implemented, use \'iterative\' or set \'parallel\' to True for multithreading.")
        else:
            func(*args, **kwargs)
    for thread in threads: thread.join()
        

#############################################################
def sub_filepath(filepath: str, subregex: str):
    m = re.search(subregex, filepath)
    if m is None: return filepath
    else: return filepath[m.start():]


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
    if split_data_mc_eras: return sorted(MC_eras), sorted(Data_eras)
    else: return sorted(MC_eras | Data_eras)


#############################################################
def check_train_filepaths(train_filepaths: list, eras: list, class_sample_map: dict):
    good_dataset_bool = True
    for glob_name in [glob_name for glob_names in class_sample_map.values() for glob_name in glob_names]:
        for era in eras:
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

def get_input_filepaths(eras: str|list[str], class_sample_map: dict, regex: str|list[str]="", check_dataset: bool=False, dataset: str='train'):
    if type(eras) is str: eras = get_era_filepaths(eras)
    input_filepaths = []
    
    for era in eras:
        sample_filepaths = eos.glob_eos(os.path.join(era, "**", regex), recursive=True)
        for sample_filepath in sample_filepaths:
            sub_sample_filepath = sample_filepath[len(era):]
            if match_sample(
                sub_sample_filepath, 
                {glob_name for glob_names in class_sample_map.values() for glob_name in glob_names}
            ) is not None or (match_sample(sub_sample_filepath, ['Data']) and dataset == "test"):
                input_filepaths.append(sample_filepath)

    if check_dataset: assert check_train_filepaths(input_filepaths, eras, class_sample_map), f"Train dataset is missing some samples for some eras."
    
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


#############################################################
# Decorators for batch handling
def batched_writer(pq_iter_func, infilepath: str, outfilepath: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
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
    return decorator

def batched_executor(pq_iter_func, filepath: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            eos_filepath = eos.load_file_eos(filepath, **kwargs)
            batched_return = []
            for pq_batch in pq_iter_func(eos_filepath, **kwargs):
                batched_return.append(func(pq_batch.to_pandas(), *args, **kwargs))
            eos.delete_lockfile(eos_filepath)
            return batched_return
        return wrapper
    return decorator


#############################################################
# Decorators for multifile handling
def multifile_executor(filepaths: list[str]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            multifile_return = []
            for filepath in filepaths:
                eos_filepath = eos.load_file_eos(filepath, **kwargs)
                multifile_return.append(func(eos_filepath, *args, **kwargs))
                eos.delete_lockfile(eos_filepath)
            return multifile_return
        return wrapper
    return decorator


#############################################################
# Decorators for multifold handling
def multifold_executor(n_folds: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            multifold_return = []
            for fold in range(n_folds):
                multifold_return.append(func(fold, *args, **kwargs))
            return multifold_return
        return wrapper
    return decorator