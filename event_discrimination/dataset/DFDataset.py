# Stdlib packages
import argparse
import copy
import glob
import json
import logging
import os
import subprocess
import sys

# Common Py packages
import numpy as np  
import pandas as pd
import pyarrow.parquet as pq

# HEP packages
from eos_utils import copy_eos

################################


from HHtobbyy.event_discrimination.preprocessing.preprocessing_utils import (
    get_era_filepaths
)
from HHtobbyy.event_discrimination.preprocessing.BDT_preprocessing_utils import (
    no_standardize, apply_logs
)
from HHtobbyy.event_discrimination.preprocessing.retrieval_utils import argsorted

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "input_eras",
    help="File for input eras to run processing"
)
parser.add_argument(
    "MODEL_config",
    help="Full filepath on cluster `.py` config file for BDT"
)
parser.add_argument(
    "output_dirpath",
    help="Full filepath on cluster for output to be dumped"
)
parser.add_argument(
    "--add_test", 
    action="store_true",
    help="Flag to extend existing parquets with extra samples for testing/evaluation (i.e. NOT training) following the same standardization -- requires there to be samples and standarization JSONs at the output_dirpath location. Only adds additional test files"
)
parser.add_argument(
    "--replace_test", 
    action="store_true",
    help="Flag to extend existing parquets with extra samples for testing/evaluation (i.e. NOT training) following the same standardization -- requires there to be samples and standarization JSONs at the output_dirpath location. Removes previous test files"
)
parser.add_argument(
    "--dont_check_dataset", 
    action="store_true",
    help="Boolean to *not* check the train dataset for the samples for each era"
)

args = parser.parse_args()
MODEL_CONFIG = args.MODEL_config.replace('.py', '').split('/')[-1]
exec(f"from {MODEL_CONFIG} import *")

################################


NECESSARY_AUX_VARIABLES = {'weight', 'eventWeight', 'sample_name', 'hash'}

################################


class DFDataset:
    def __init__(filepaths, model_vars: set, aux_vars: set):







def get_df_mask(df):
    if DF_MASK == 'none':
        return (df['pt'] > 0)
    if DF_MASK == 'default':
        return (df[f'{JET_PREFIX}_resolved_BDT_mask'] > 0)
    else:
        raise NotImplementedError(f"Mask method {DF_MASK} not yet implemented, use \'default\'.")

def get_dfs(filepaths, BDT_vars, AUX_vars):
    dfs, aux_dfs = {}, {}
    for filepath in sorted(filepaths):
        pq_file = pq.ParquetFile(filepath)
        for pq_batch in pq_file.iter_batches(batch_size=524_288, columns=BDT_vars+AUX_vars):
            df_batch = pq_batch.to_pandas()
            df_mask = get_df_mask(df_batch)
            if filepath not in dfs:
                dfs[filepath] = df_batch.loc[df_mask, BDT_vars].reset_index(drop=True)
                aux_dfs[filepath] = df_batch.loc[df_mask, AUX_vars].reset_index(drop=True)
            else:
                dfs[filepath] = pd.concat([dfs[filepath], df_batch.loc[df_mask, BDT_vars].reset_index(drop=True)]).reset_index(drop=True)
                aux_dfs[filepath] = pd.concat([aux_dfs[filepath], df_batch.loc[df_mask, AUX_vars].reset_index(drop=True)]).reset_index(drop=True)

    return dfs, aux_dfs

def get_split_dfs(filepaths, BDT_vars, AUX_vars, fold_idx):
    # Train/Val events are those with eventID % mod_val != fold, test events are the others
    dfs, aux_dfs = get_dfs(filepaths, BDT_vars, AUX_vars)

    train_dfs, train_aux_dfs, test_dfs, test_aux_dfs = {}, {}, {}, {}
    for filepath in sorted(filepaths):
        train_mask = (aux_dfs[filepath]['event'] % TRAIN_MOD).ne(fold_idx)
        test_mask = (aux_dfs[filepath]['event'] % TRAIN_MOD).eq(fold_idx)

        train_dfs[filepath] = dfs[filepath].loc[train_mask].reset_index(drop=True)
        train_aux_dfs[filepath] = aux_dfs[filepath].loc[train_mask].reset_index(drop=True)
        test_dfs[filepath] = dfs[filepath].loc[test_mask].reset_index(drop=True)
        test_aux_dfs[filepath] = aux_dfs[filepath].loc[test_mask].reset_index(drop=True)
        
    return train_dfs, train_aux_dfs, test_dfs, test_aux_dfs

def compute_standardization(train_dfs, train_dfs_fold):
    merged_train_df = pd.concat(list(train_dfs.values())+list(train_dfs_fold.values()), ignore_index=True)

    merged_train_df = merged_train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    merged_train_df = apply_logs(merged_train_df)
    masked_x_sample = np.ma.array(merged_train_df, mask=(merged_train_df == FILL_VALUE))

    x_mean = masked_x_sample.mean(axis=0)
    x_std = masked_x_sample.std(axis=0)
    for i, col in enumerate(merged_train_df.columns):
        if no_standardize(col):
            x_mean[i] = 0
            x_std[i] = 1

    return x_mean, x_std

def preprocess_resolved_bdt(input_filepaths, output_dirpath):
    """
    Builds and standardizes the dataframes to be used for training,

    Inputs:
        - input_filepaths = {
            'train-test': ['', '', '', ...]
            'train': ['','', ...]
            'test': ['','', ...]
        }
        - output_dirpath = <str> filepath to dump output (defaults to cwd)
    """
    # Defining class definitions for samples #
    class_sample_map_filepath = os.path.join(output_dirpath, 'class_sample_map.json')
    if class_sample_map_filepath.split('/')[1] == 'eos':
        tmp_file = f"tmp_map{hash(output_dirpath)}.json"
        with open(tmp_file, 'w') as f: json.dump(CLASS_SAMPLE_MAP, f)
        copy_eos(tmp_file, 'root://cmseos.fnal.gov/'+'/'.join(['']+class_sample_map_filepath.split('/')[3:]), force=True)
        subprocess.run(['rm', tmp_file])
    else:
        with open(class_sample_map_filepath, 'w') as f: json.dump(CLASS_SAMPLE_MAP, f)

    # Defining variables to use #
    assert len(NECESSARY_AUX_VARIABLES & AUX_VARIABLES) == len(NECESSARY_AUX_VARIABLES), f"Missing some necessary AUX variables, see \"NECESSARY_AUX_VARIABLES\" for list"
    MODEL_variables, AUX_variables = sorted(MODEL_variables), sorted(AUX_VARIABLES)
    
    train_dfs, train_aux_dfs = get_dfs(input_filepaths['train'], MODEL_variables, AUX_variables)
    test_dfs, test_aux_dfs = get_dfs(input_filepaths['test'], MODEL_variables, AUX_variables)

    for fold_idx in range(TRAIN_MOD):
        (
            train_dfs_fold, train_aux_dfs_fold, 
            test_dfs_fold, test_aux_dfs_fold 
        ) = get_split_dfs(input_filepaths['train-test'], MODEL_variables, AUX_variables, fold_idx)

        stdjson_filepath = os.path.join(output_dirpath, 'standardization.json')
        if not ADD_TEST:
            x_mean, x_std = compute_standardization(train_dfs, train_dfs_fold)
            stdjson = {'col': MODEL_variables, 'mean': x_mean.tolist(), 'std': x_std.tolist()}
            if stdjson_filepath.split('/')[1] == 'eos':
                tmp_file = f"tmp_std{hash(output_dirpath)}.json"
                with open(tmp_file, 'w') as f: json.dump(stdjson, f)
                copy_eos(tmp_file, 'root://cmseos.fnal.gov/'+'/'.join(['']+stdjson_filepath.split('/')[3:]), force=True)
                subprocess.run(['rm', tmp_file])
            else:
                with open(stdjson_filepath, 'w') as f: json.dump(stdjson, f)
        else:
            with open(stdjson_filepath, 'r') as f:
                stdjson = json.load(f)
            if len(stdjson['col']) != len(MODEL_variables):
                raise Exception(f"Mismatch between number of new variables being used ({len(MODEL_variables)}) and number of variables in dataset ({len(stdjson['col'])}), check `standardization.json` file.")
            sort_indices = argsorted(stdjson['col'])
            if any(sorted(stdjson['col'])[i] != MODEL_variables[i] for i in range(len(MODEL_variables))): 
                raise Exception("Mismatch between new variables and variables in dataset, check `standardization.json` file.")
            x_mean, x_std = [stdjson['mean'][i] for i in sort_indices], [stdjson['std'][i] for i in sort_indices]


        if not ADD_TEST:
            for filepath in train_dfs.keys():
                train_dfs_fold[filepath] = copy.deepcopy(train_dfs[filepath])
                train_aux_dfs_fold[filepath] = copy.deepcopy(train_aux_dfs[filepath])

            for file_i, (filepath, df) in enumerate(train_dfs_fold.items()):
                output_filepath = make_output_filepath(filepath, output_dirpath, f"train{fold_idx}")

                cols = list(df.columns)
                df = apply_logs(df)
                df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
                df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

                for aux_col in train_aux_dfs_fold[filepath].columns:
                    df[f"AUX_{aux_col}"] = train_aux_dfs_fold[filepath].loc[:,aux_col]

                if output_filepath.split('/')[1] == 'eos':
                    tmp_file = f"tmp{hash(output_filepath)}.parquet"
                    df.to_parquet(tmp_file)
                    copy_eos(tmp_file, 'root://cmseos.fnal.gov/'+'/'.join(['']+output_filepath.split('/')[3:]), force=True)
                    subprocess.run(['rm', tmp_file])
                else:
                    df.to_parquet(output_filepath)


        for filepath in test_dfs.keys():
            test_dfs_fold[filepath] = copy.deepcopy(test_dfs[filepath])
            test_aux_dfs_fold[filepath] = copy.deepcopy(test_aux_dfs[filepath])

        for file_i, (filepath, df) in enumerate(test_dfs_fold.items()):
            output_filepath = make_output_filepath(filepath, output_dirpath, f"test{fold_idx}")
            prev_test_filepaths = glob.glob(
                output_filepath[:output_filepath.rfind(f"test{fold_idx}")+len(f"test{fold_idx}")]
                + "*.parquet"
            )
            if len(prev_test_filepaths) > 0 and ADD_TEST:
                logger.log(1, f"{filepath[filepath.find(BASE_FILEPATH):filepath.rfind('/')]} skipped because test file already exists and \'--add_test\' option selected")
                continue
            else: print(filepath)

            cols = list(df.columns)
            df = apply_logs(df)
            df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
            df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

            for aux_col in test_aux_dfs_fold[filepath].columns:
                df[f"AUX_{aux_col}"] = test_aux_dfs_fold[filepath].loc[:,aux_col]

            if output_filepath.split('/')[1] == 'eos':
                tmp_file = f"tmp{hash(output_filepath)}.parquet"
                df.to_parquet(tmp_file)
                copy_eos(tmp_file, 'root://cmseos.fnal.gov/'+'/'.join(['']+output_filepath.split('/')[3:]), force=True)
                subprocess.run(['rm', tmp_file])
            else:
                df.to_parquet(output_filepath)

if __name__ == '__main__':
    print('='*60)
    print(f'Starting Resolved BDT processing at {CURRENT_TIME}')

    ADD_TEST = args.add_test
    args_output_dirpath = os.path.normpath(args.output_dirpath)
    if ADD_TEST: CURRENT_TIME = args_output_dirpath[-len('YYYY-MM-DD_HH-MM-SS'):]
    else: args_output_dirpath = os.path.join(args_output_dirpath, f"{DATASET_TAG}_{CURRENT_TIME}")
    if not os.path.exists(args_output_dirpath):
        os.makedirs(args_output_dirpath)
    ERAS = get_era_filepaths(args.input_eras)
    input_filepaths = get_input_filepaths()

    preprocess_resolved_bdt(input_filepaths, args_output_dirpath)
    print(f'Finished Resolved BDT processing')
