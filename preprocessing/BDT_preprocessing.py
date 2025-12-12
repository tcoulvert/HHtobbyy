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

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "plotting/"))

from preprocessing_utils import (
    match_sample, match_regex, get_era_filepaths
)
from BDT_preprocessing_utils import (
    no_standardize, apply_logs
)
from retrieval_utils import argsorted
# from plot_vars import plot_vars

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "input_eras",
    help="File for input eras to run processing"
)
parser.add_argument(
    "BDT_config",
    help="Full filepath on cluster `.py` config file for BDT"
)
parser.add_argument(
    "output_dirpath",
    help="Full filepath on cluster for output to be dumped"
)
parser.add_argument(
    "--remake_test", 
    action="store_true",
    help="Flag to extend existing parquets with extra samples for testing/evaluation (i.e. NOT training) following the same standardization -- requires there to be samples and standarization JSONs at the output_dirpath location."
)
parser.add_argument(
    "--plots", 
    action="store_true",
    help="Makes plots of dataset input variables"
)
parser.add_argument(
    "--debug", 
    action="store_true",
    help="Flag to print debug messages"
)
parser.add_argument(
    "--dryrun", 
    action="store_true",
    help="Flag to not save parquets out and just try running"
)
parser.add_argument(
    "--dont_check_dataset", 
    action="store_true",
    help="Boolean to *not* check the train dataset for the samples for each era"
)

args = parser.parse_args()
BDT_CONFIG = args.BDT_config.replace('.py', '').split('/')[-1]
exec(f"from {BDT_CONFIG} import *")

################################


def check_train_dataset(train_filepaths: list):
    good_dataset_bool = True
    for glob_name in [glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names]:
        for era in ERAS:
            if match_regex(f"{era}*{glob_name}", train_filepaths) is None:
                good_dataset_bool = False; break 
    return good_dataset_bool

def get_input_filepaths():
    input_filepaths = {'train-test': list(), 'train': list(), 'test': list()}
    
    for era in ERAS:
        sample_filepaths = glob.glob(os.path.join(era, "**", f"*{END_FILEPATH}"), recursive=True)
        for sample_filepath in sample_filepaths:
            if (
                match_sample(sample_filepath, TEST_ONLY_SAMPLES) is not None
                and match_sample(sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is None
            ):
                input_filepaths['test'].append(sample_filepath)
            elif (
                match_sample(sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is not None 
                and match_sample(sample_filepath, TRAIN_ONLY_SAMPLES) is not None
            ):
                input_filepaths['train'].append(sample_filepath)
            elif match_sample(sample_filepath, {glob_name for glob_names in CLASS_SAMPLE_MAP.values() for glob_name in glob_names}) is not None:
                input_filepaths['train-test'].append(sample_filepath)
            else:
                if DEBUG:
                    logger.warning(f"{sample_filepath} \nSample not found in any dict (TRAIN_TEST_SAMPLES, TRAIN_ONLY_SAMPLES, TEST_ONLY_SAMPLES). Continuing with other samples.")
                continue

    if not args.dont_check_dataset:
        assert check_train_dataset(input_filepaths['train']), f"Train dataset is missing some samples for some eras."
    
    return input_filepaths

def make_output_filepath(filepath, base_output_dirpath, extra_text):
    filename = filepath[filepath.rfind('/')+1:]
    output_dirpath = os.path.join(
        base_output_dirpath,
        filepath[filepath.find(BASE_FILEPATH):filepath.rfind('/')]
    )
    if not os.path.exists(output_dirpath) and not DRYRUN:
        os.makedirs(output_dirpath)

    filename = filename[:filename.rfind('.')] + f"_{extra_text}_{CURRENT_TIME}" + filename[filename.rfind('.'):]

    return os.path.join(output_dirpath, filename)

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
        print(filepath)
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
    if not DRYRUN:
        class_sample_map_filepath = os.path.join(output_dirpath, 'class_sample_map.json')
        with open(class_sample_map_filepath, 'w') as f:
            json.dump(CLASS_SAMPLE_MAP, f)

    # Defining variables to use #
    BDT_variables, AUX_variables = sorted(BDT_VARIABLES), sorted(AUX_VARIABLES)
    
    train_dfs, train_aux_dfs = get_dfs(input_filepaths['train'], BDT_variables, AUX_variables)
    test_dfs, test_aux_dfs = get_dfs(input_filepaths['test'], BDT_variables, AUX_variables)

    for fold_idx in range(TRAIN_MOD):
        (
            train_dfs_fold, train_aux_dfs_fold, 
            test_dfs_fold, test_aux_dfs_fold 
        ) = get_split_dfs(input_filepaths['train-test'], BDT_variables, AUX_variables, fold_idx)


        stdjson_filepath = os.path.join(output_dirpath, 'standardization.json')
        if not REMAKE_TEST:
            x_mean, x_std = compute_standardization(train_dfs, train_dfs_fold)
            if not DRYRUN:
                stdjson = {'col': BDT_variables, 'mean': x_mean.tolist(), 'std': x_std.tolist()}
                with open(stdjson_filepath, 'w') as f:
                    json.dump(stdjson, f)
        else:
            with open(stdjson_filepath, 'r') as f:
                stdjson = json.load(f)
            if len(stdjson['col']) != len(BDT_variables):
                raise Exception(f"Mismatch between number of new variables being used ({len(BDT_variables)}) and number of variables in dataset ({len(stdjson['col'])}), check `standardization.json` file.")
            sort_indices = argsorted(stdjson['col'])
            if any(sorted(stdjson['col'])[i] != BDT_variables[i] for i in range(len(BDT_variables))): 
                raise Exception("Mismatch between new variables and variables in dataset, check `standardization.json` file.")
            x_mean, x_std = [stdjson['mean'][i] for i in sort_indices], [stdjson['std'][i] for i in sort_indices]


        if not REMAKE_TEST:
            for filepath in train_dfs.keys():
                train_dfs_fold[filepath] = copy.deepcopy(train_dfs[filepath])
                train_aux_dfs_fold[filepath] = copy.deepcopy(train_aux_dfs[filepath])

            for filepath, df in train_dfs_fold.items():
                output_filepath = make_output_filepath(filepath, output_dirpath, f"train{fold_idx}")
                if MAKE_PLOTS: 
                    plot_vars(
                        df, 
                        "/".join(output_filepath.split("/")[:-1]), 
                        train_aux_dfs_fold[filepath]["sample_name"][0], 
                        title=f"pre-std, train{fold_idx}"
                    )
                if DEBUG:
                    print('-'*60)
                    print(f"input = \n{filepath}\n{'-'*60}\noutput = \n{output_filepath}")
                    print(f"num events = {len(df)}")
                    print(f"sum of weights = {train_aux_dfs_fold[filepath].loc[:,'weight'].sum()}")
                    print(f"sum of eventWeights = {train_aux_dfs_fold[filepath].loc[:,'eventWeight'].sum()}")

                cols = list(df.columns)
                df = apply_logs(df)
                df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
                df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

                if MAKE_PLOTS: plot_vars(
                    df, 
                    "/".join(output_filepath.split("/")[:-1]), 
                    train_aux_dfs_fold[filepath]["sample_name"][0], 
                    title=f"post-std, train{fold_idx}"
                )

                for aux_col in train_aux_dfs_fold[filepath].columns:
                    df[f"AUX_{aux_col}"] = train_aux_dfs_fold[filepath].loc[:,aux_col]

                if not DRYRUN: df.to_parquet(output_filepath)


        for filepath in test_dfs.keys():
            test_dfs_fold[filepath] = copy.deepcopy(test_dfs[filepath])
            test_aux_dfs_fold[filepath] = copy.deepcopy(test_aux_dfs[filepath])

        for filepath, df in test_dfs_fold.items():
            output_filepath = make_output_filepath(filepath, output_dirpath, f"test{fold_idx}")
            if MAKE_PLOTS: 
                plot_vars(
                    df, 
                    "/".join(output_filepath.split("/")[:-1]), 
                    test_aux_dfs_fold[filepath]["sample_name"][0], 
                    title=f"pre-std, test{fold_idx}"
                )
            if DEBUG:
                print('-'*60)
                print(f"input = \n{filepath}\n{'-'*60}\noutput = \n{output_filepath}")
                print(f"num events = {len(df)}")
                print(f"sum of weights = {test_aux_dfs_fold[filepath].loc[:,'weight'].sum()}")
                print(f"sum of eventWeights = {test_aux_dfs_fold[filepath].loc[:,'eventWeight'].sum()}")

            cols = list(df.columns)
            df = apply_logs(df)
            df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
            df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

            if MAKE_PLOTS: plot_vars(
                df, 
                "/".join(output_filepath.split("/")[:-1]), 
                test_aux_dfs_fold[filepath]["sample_name"][0], 
                title=f"post-std, test{fold_idx}"
            )

            for aux_col in test_aux_dfs_fold[filepath].columns:
                df[f"AUX_{aux_col}"] = test_aux_dfs_fold[filepath].loc[:,aux_col]

            if not DRYRUN: df.to_parquet(output_filepath)

if __name__ == '__main__':
    print('='*60)
    print(f'Starting Resolved BDT processing at {CURRENT_TIME}')

    DEBUG = args.debug
    DRYRUN = args.dryrun
    MAKE_PLOTS = args.plots
    REMAKE_TEST = args.remake_test
    args_output_dirpath = os.path.normpath(args.output_dirpath)
    if REMAKE_TEST: CURRENT_TIME = args_output_dirpath[args_output_dirpath.rfind('/')+1:]
    else: args_output_dirpath = os.path.join(args_output_dirpath, f"{DATASET_TAG}_{CURRENT_TIME}")
    if not os.path.exists(args_output_dirpath) and not DRYRUN:
        os.makedirs(args_output_dirpath)
    ERAS = get_era_filepaths(args.input_eras)
    input_filepaths = get_input_filepaths()

    preprocess_resolved_bdt(input_filepaths, args_output_dirpath)
    print(f'Finished Resolved BDT processing')
