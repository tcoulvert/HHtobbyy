# Stdlib packages
import argparse
import datetime
import logging
import os

# Common Py packages
import numpy as np  
import pandas as pd
import pyarrow.parquet as pq

# HEP packages
from eos_utils import save_file_eos, load_file_eos

# Workspace packages
from HHtobbyy.event_discrimination.preprocessing.BDT_preprocessing_utils import (
    no_standardize, apply_logs
)
from HHtobbyy.event_discrimination.dataset.DFDataset_utils import make_output_filepath
from HHtobbyy.workspace.retrieval_utils import FILL_VALUE

# MODEL_CONFIG = args.MODEL_config.replace('.py', '').split('/')[-1]
# exec(f"from {MODEL_CONFIG} import *")


class DFDataset:
    def __init__(config: dict, output_dirpath: str=''):
        self.xrd_redirector = config['xrd_redirector']  # 'root://cmseos.fnal.gov/'
        if output_dirpath == '':
            self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
            self.output_dirpath = os.path.normpath(os.path.join(config['output_dirpath'], f"{config['dataset_tag']}_{self.current_time}"))
            if not os.path.exists(self.output_dirpath):
                os.makedirs(self.output_dirpath)

            config_filepath = os.path.join(self.output_dirpath, f'dataset_config.json')
            save_file_eos(config, config_filepath, self.xrd_redirector)
        else:
            self.output_dirpath = output_dirpath

        self.pq_batch_size = config['pq_batch_size']  # 524_288
        self.seed = config['seed']

        self.model_vars = sorted(config['model_vars'])
        self.aux_vars = sorted(config['aux_vars'])
        self.all_vars = sorted(self.model_vars + self.aux_vars)
        self.new_aux_vars = sorted(['AUX_' + aux_var for aux_var in self.aux_vars])
        self.new_all_vars = sorted(self.model_vars + self.new_aux_vars)
        self.necessary_aux_vars = {'weight', 'eventWeight', 'sample_name', 'hash'}

        self.mask_var = config['mask_var']

        self.n_folds = config['n_folds']

        self.standardization_method = config['standardization_method']

    def presel_mask(self, df: pd.DataFrame):
        if self.mask_var == 'none': return np.ones(len(df), dtype=bool)
        elif self.mask_var in df.columns: return np.asarray(df[self.mask_var] > 0, dtype=bool)

    def train_mask(self, df: pd.DataFrame, fold: int):
        return np.asarray(df['AUX_event'].mod(self.n_folds).ne(fold), dtype=bool)
    def test_mask(self, df: pd.DataFrame, fold: int):
        return ~self.train_mask(df, fold)
    
    def make_df(self, filepath: str):
        pq_file = pq.ParquetFile(filepath)
        assert all(necessary_aux_var in pq_file.schema.keys() for necessary_aux_var in self.necessary_aux_vars), f"ERROR: Required to have all the necessary aux vars {self.necessary_aux_vars} present for downstream processing and tracking. Currently missing {self.necessary_aux_vars - set(pq_file.schema.keys())}"
        df = pd.DataFrame(columns=self.new_all_vars).astype([value for key, value in pq_file.schema.items() if key in self.all_vars])
        for pq_batch in pq_file.iter_batches(batch_size=self.pq_batch_size, columns=list(set(self.all_vars))):
            df_batch = pq_batch.to_pandas()
            mask = self.presel_mask(df_batch)
            df_batch = pd.merge(df_batch.loc[:, self.model_vars], df_batch.loc[:, self.aux_vars].rename(dict(zip(self.aux_vars, self.new_aux_vars))), how='outer')
            df = pd.concat([df, df_batch.loc[mask].reset_index(drop=True)])

    def compute_standardization(self, train_dfs: dict[str, pd.DataFrame], fold: int):
        merged_train_df = pd.concat([df.loc[:, self.model_vars] for df in train_dfs.values()]).reset_index(drop=True)
        if self.standardization_method.lower() == 'zscore': compute_zscore_standardization(merged_train_df, fold)
        else: raise NotImplementedError(f"Standardization method not yet implemented, use \'zscore\'.")
    def compute_zscore_standardization(self, merged_train_df: pd.DataFrame, fold: int):
        merged_train_df = merged_train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        merged_train_df = apply_logs(merged_train_df)
        masked_x_sample = np.ma.array(merged_train_df, mask=(merged_train_df == FILL_VALUE))

        x_mean = masked_x_sample.mean(axis=0)
        x_std = masked_x_sample.std(axis=0)
        for i, col in enumerate(self.model_vars):
            if no_standardize(col):
                x_mean[i] = 0; x_std[i] = 1

        zscore_std = {'col': self.model_vars, 'mean': x_mean.tolist(), 'std': x_std.tolist()}
        zscore_std_filepath = os.path.join(self.output_dirpath, f'zscore_standardization_fold{fold}.json')
        save_file_eos(zscore_std, zscore_std_filepath, self.xrd_redirector)

    def apply_standardization(self, df: pd.DataFrame, fold: int):
        slimmed_df = df.loc[:, self.model_vars]
        if self.standardization_method.lower() == 'zscore': apply_zscore_standardization(slimmed_df, fold)
        else: raise NotImplementedError(f"Standardization method not yet implemented, use \'zscore\'.")
        return pd.concat([slimmed_df, df.loc[:, self.new_aux_vars]])
    def apply_zscore_standardization(self, df: pd.DataFrame, fold: int):
        zscore_std_filepath = os.path.join(self.output_dirpath, f'zscore_standardization_fold{fold}.json')
        zscore_std = load_file_eos(zscore_std_filepath, self.xrd_redirector)
        
        df = apply_logs(df)
        df = (np.ma.array(df, mask=(df == FILL_VALUE)) - zscore_std['mean']) / zscore_std['std']
        df = pd.DataFrame(df.filled(FILL_VALUE), columns=zscore_std['col'])


    def make_train(self, filepaths: list, fold: int):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"

        dfs = {}
        for filepath in filepaths:
            dfs[filepath] = self.make_df(filepath)
            mask = self.train_mask(dfs[filepath], fold)
            dfs[filepath] = dfs[filepath].loc[mask].reset_index(drop=True)

        self.compute_standardization(dfs)

        for filepath in filepaths:
            standardized_df = self.apply_standardization(dfs[filepath], fold)
            save_file_eos(standardized_df, make_output_filepath(filepath, self.output_dirpath, f"train{fold}"))

    def make_test(self, filepaths: list, fold: int, force: bool=False):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"

        for filepath in filepaths:
            df = self.make_df(filepath)
            mask = self.testmask(df, fold)
            df = df.loc[mask].reset_index(drop=True)
            
            standardized_df = self.apply_standardization(df, fold)
            save_file_eos(standardized_df, make_output_filepath(filepath, self.output_dirpath, f"test{fold}"), force=force)
