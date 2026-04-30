# Stdlib packages
import glob
import os
import re
from threading import Thread 

################################


FILL_VALUE = -999

#############################################################
def multifold(func, args, n_folds, parallel: bool=False, condor: dict={}):
    threads = []
    for fold in range(n_folds):
        arg = (fold, )+args
        if parallel: 
            thread = Thread(target=func, name=f"Fold {fold}", args=arg)
            thread.start(); threads.append(thread)
        elif condor != {}:
            raise NotImplementedError(f"Multifold via Condor not yet implemented, use \'iterative\' or set \'parallel\' to True for multithreading.")
        else:
            func(*arg)
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

def get_input_filepaths(eras: str|list[str], class_sample_map: dict, regex: str|list[str]="", check_dataset: bool=False):
    if type(eras) is str: eras = get_era_filepaths(eras)
    input_filepaths = []
    
    for era in eras:
        print('-'*60, era)
        sample_filepaths = glob.glob(os.path.join(era, "**", regex), recursive=True)
        for sample_filepath in sample_filepaths:
            sub_sample_filepath = sample_filepath[len(era):]
            if match_sample(
                sub_sample_filepath, 
                {glob_name for glob_names in class_sample_map.values() for glob_name in glob_names}
            ) is not None or match_sample(sub_sample_filepath, ['Data']):
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
