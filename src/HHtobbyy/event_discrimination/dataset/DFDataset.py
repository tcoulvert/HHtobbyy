# Stdlib packages
import datetime
import os

# Common Py packages
import numpy as np  
import pandas as pd
import pyarrow.parquet as pq

# ML packages
from sklearn.model_selection import train_test_split

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.dataset.DFDataset_utils import (
    no_standardize, apply_logs, map_filepath_to_class
)
from HHtobbyy.event_discrimination.dataset.DFDataset_utils import make_output_filepath
from HHtobbyy.workspace_utils.retrieval_utils import (
    FILL_VALUE, get_traintest_filepaths_func, get_test_filepaths_func, match_regex
)


class DFDataset:
    """
    Defines the pandas DataFrame format for the dataset. DataFrame is useful as a common
    format that can easily be converted downstream to the model-specific format (see ModelDataset).
    """

    def __init__(self, config: str|dict={}, output_dirpath: str=''):
        """
        Arguments
        ----------
        config : dict
            Configuration for the dataset
        output_dirpath : str
            Overrides config value for dirpath, used if loading dataset rather than making a new one
        ----------
        NOTE: At least one argument is required.
        """

        # Filenames for common retrieval
        self.config_filename = 'dataset_config.json'
        self.standardization_subfilename = 'standardization_fold'
        self.class_sample_map_filename = 'class_sample_map.json'

        # Current time of execution
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'

        assert config != {} or output_dirpath != '', f"ERROR: At least one argument is required."
        if type(config) is str: 
            config = eos.load_file_eos(dict, config)
        elif config == {}:
            config = eos.load_file_eos(dict, os.path.join(output_dirpath, self.config_filename))
            self.current_time = os.path.normpath(output_dirpath).split('_')[-1]

        # Dirpath to dump dataset
        self.output_dirpath = output_dirpath

        # Batch size for loading parquets
        self.pq_batch_size = 524_288

        # RNG seed
        self.seed = 21

        # Mask variable for preselection, 'none' for no extra pre-selection
        self.mask_var = 'none'

        # Number of folds, one model per fold
        self.n_folds = 5

        # Fraction of training data to use for validation, if using
        self.val_split = 0.2

        # Method to split train and val
        self.train_val_split_method == 'scikit'

        # Method used for the standardization
        self.standardization_method = 'zscore'

        # Standard prefix for auxiliary variables, i.e. variables useful 
        #   for event-identification but *not* used in the training
        self.aux_var_prefix = 'AUX_'

        # End of filepath for files to pull using eras selection
        #  (doesn't matter if passing filepaths directly)
        self.filepostfix = 'preprocessed.parquet'

        # Basic fileprefix to separate local machine directories from HiggsDNA 
        #   (or equivalent preprocessor) directories
        self.base_filepath = 'Run3_20'

        # Optional dictionaries to perform reweighting for training and testing
        self.test_sample_reweighting = 'none'
        self.train_sample_reweighting = 'none'
        self.train_class_reweighting = 'none'
        self.normalize_signal_to_bkg = True

        # Processes the config
        self.process_config(config)


    #############################################################
    # Configuration preocessing
    def process_config(self, config: dict):
        reqd_keys = ['output_dirpath', 'dataset_tag', 'model_vars', 'aux_vars', 'class_sample_map']
        assert all(key in config.keys() for key in reqd_keys), f"Config file required to have some variables: {reqd_keys}, received config is missing {set(config.keys()) - set(reqd_keys)}"

        def make_output_dirpath():
            if self.output_dirpath == '':
                self.output_dirpath = os.path.normpath(os.path.join(config['output_dirpath'], f"{config['dataset_tag']}_{self.current_time}"))
                os.makedirs(self.output_dirpath, exist_ok=True)

        def make_var_lists():
            self.model_vars = sorted(self.model_vars)
            self.aux_vars = sorted(self.aux_vars)
            self.all_vars = sorted(self.model_vars + self.aux_vars)
            self.new_aux_vars = sorted([self.aux_var_prefix + aux_var for aux_var in self.aux_vars])
            self.new_all_vars = sorted(self.model_vars + self.new_aux_vars)
            self.necessary_aux_vars = {'weight', 'eventWeight', 'sample_name', 'hash'}

        for key, value in config.items():
            if key == 'output_dirpath': make_output_dirpath()
            else: 
                if hasattr(self, key): setattr(self, key, value)
        make_var_lists()

        # Number of classes
        self.n_classes = len(self.class_sample_map)

        class_sample_map_filepath = os.path.join(self.output_dirpath, self.class_sample_map_filename)
        try: eos.load_file_eos(dict, class_sample_map_filepath)
        except: eos.save_file_eos(self.class_sample_map, class_sample_map_filepath)

        config_filepath = os.path.join(self.output_dirpath, self.config_filename)
        try: eos.load_file_eos(dict, config_filepath)
        except: eos.save_file_eos(config, config_filepath)


    #############################################################
    # Event masking
    def presel_mask(self, df: pd.DataFrame):
        if self.mask_var == 'none': return np.ones(len(df), dtype=bool)
        elif self.mask_var in df.columns: return np.asarray(df[self.mask_var] > 0, dtype=bool)

    def train_mask(self, df: pd.DataFrame, fold: int):
        return np.asarray(df['AUX_event'].mod(self.n_folds).ne(fold), dtype=bool)
    def test_mask(self, df: pd.DataFrame, fold: int):
        return ~self.train_mask(df, fold)
    

    #############################################################
    # Basic DF build
    def make_df(self, filepath: str):
        pq_file = pq.ParquetFile(filepath)
        assert all(necessary_aux_var in pq_file.schema.keys() for necessary_aux_var in self.necessary_aux_vars), f"ERROR: Required to have all the necessary aux vars {self.necessary_aux_vars} present for downstream processing and tracking. Currently missing {self.necessary_aux_vars - set(pq_file.schema.keys())}"
        df = pd.DataFrame(columns=self.new_all_vars).astype([value for key, value in pq_file.schema.items() if key in self.all_vars])
        for pq_batch in pq_file.iter_batches(batch_size=self.pq_batch_size, columns=list(set(self.all_vars))):
            df_batch = pq_batch.to_pandas()
            mask = self.presel_mask(df_batch)
            df_batch = pd.merge(df_batch.loc[:, self.model_vars], df_batch.loc[:, self.aux_vars].rename(dict(zip(self.aux_vars, self.new_aux_vars))), how='outer')
            df = pd.concat([df, df_batch.loc[mask].reset_index(drop=True)])


    #############################################################
    # Additional variables
    def sample_reweighting(self, df: pd.DataFrame, sample_reweight: str|dict, reweight_var: str):
        if type(sample_reweight) is str and sample_reweight == 'none': return
        elif type(sample_reweight) is dict:
            df_unique_sample_tags = df[f'{self.aux_var_prefix}sample_name'].unique()
            for sample_tag, reweight in sample_reweight.items():
                df_sample_tag = match_regex(sample_tag, df_unique_sample_tags)
                if df_sample_tag is not None: 
                    df.loc[df[f'{self.aux_var_prefix}sample_name'].eq(df_sample_tag), reweight_var] *= reweight
        else: raise NotImplementedError(f"Reweight method not yet implemented, use \'none\' or pass a dict.")
    def class_reweighting(self, df: pd.DataFrame, class_reweight: str|dict, reweight_var: str):
        if type(class_reweight) is str and class_reweight == 'none': return
        elif type(class_reweight) is dict:
            assert set(class_reweight.keys()).issubset(set(self.class_sample_map.keys())), f"ERROR: Input class_reweight dictionary has target classes not in the class_sample_map: {set(class_reweight.keys()) - set(self.class_sample_map.keys())}"
            assert set([value[1] for value in class_reweight.values()]).issubset(set(self.class_sample_map.keys())), f"ERROR: Input class_reweight dictionary has reference classes not in the class_sample_map: {set(class_reweight.keys()) - set(self.class_sample_map.keys())}"
            sample_reweight = {
                sample_tag: reweight * (
                    df.loc[df[f'{self.aux_var_prefix}label1D'].isin([i for i, ref_class_tag in enumerate(self.class_sample_map.keys()) if ref_class_tag in ref_class_tags]), reweight_var].sum()
                    / df.loc[df[f'{self.aux_var_prefix}label1D'].eq(list(self.class_sample_map).index(class_tag)), reweight_var].sum()
                )
                for class_tag, (reweight, ref_class_tags) in class_reweight.items() 
                for sample_tag in self.class_sample_map[class_tag]
            }
            self.sample_reweighting(df, sample_reweight, reweight_var)
        else: raise NotImplementedError(f"Reweight method not yet implemented, use \'none\' or pass a dict.")

    def add_vars(self, df: pd.DataFrame, class_idx: int):
        df[f'{self.aux_var_prefix}label1D'] = class_idx

        # Reweighting eventWeight if improperly preprocessed
        self.sample_reweighting(df, self.test_sample_reweighting, f'{self.aux_var_prefix}eventWeight')

        # Creating and reweighting eventWeightTrain for training
        df[f'{self.aux_var_prefix}eventWeightTrain'] = df[f'{self.aux_var_prefix}eventWeight']
        self.sample_reweighting(df, self.train_sample_reweighting, f'{self.aux_var_prefix}eventWeightTrain')


    #############################################################
    # Standardization
    def compute_standardization(self, train_dfs: dict[str, pd.DataFrame], fold: int):
        merged_train_df = pd.concat([df.loc[:, self.model_vars] for df in train_dfs.values()]).reset_index(drop=True)
        if self.standardization_method.lower() == 'zscore': self.compute_zscore_standardization(merged_train_df, fold)
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
        zscore_std_filepath = os.path.join(self.output_dirpath, f'zscore_{self.standardization_subfilename}{fold}.json')
        eos.save_file_eos(zscore_std, zscore_std_filepath)

    def apply_standardization(self, df: pd.DataFrame, fold: int):
        slimmed_df = df.loc[:, self.model_vars]
        if self.standardization_method.lower() == 'zscore': self.apply_zscore_standardization(slimmed_df, fold)
        else: raise NotImplementedError(f"Standardization method not yet implemented, use \'zscore\'.")
        return pd.concat([slimmed_df, df.loc[:, self.new_aux_vars]])
    def apply_zscore_standardization(self, df: pd.DataFrame, fold: int):
        zscore_std_filepath = os.path.join(self.output_dirpath, f'zscore_{self.standardization_subfilename}{fold}.json')
        zscore_std = eos.load_file_eos(dict, zscore_std_filepath)
        
        df = apply_logs(df)
        df = (np.ma.array(df, mask=(df == FILL_VALUE)) - zscore_std['mean']) / zscore_std['std']
        df = pd.DataFrame(df.filled(FILL_VALUE), columns=zscore_std['col'])


    #############################################################
    # Train/Val splitting
    def train_val_split(self):
        if self.train_val_split_method == 'scikit': return train_test_split
        else: raise NotImplementedError(f"Train/Val split method not yet implemented, use \'scikit\'.")


    #############################################################
    # Building
    def make_all_train(self, filepaths: list):
        for fold in range(self.n_folds): self.make_train(filepaths, fold)
    def make_train(self, filepaths: list, fold: int):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"

        dfs = {}
        for filepath in filepaths:
            dfs[filepath] = self.make_df(filepath)
            mask = self.train_mask(dfs[filepath], fold)
            dfs[filepath] = dfs[filepath].loc[mask].reset_index(drop=True)
            self.add_vars(dfs[filepath], fold, map_filepath_to_class(self.class_sample_map, filepath[filepath.find(self.base_filepath):]))

        self.compute_standardization(dfs)

        self.class_reweighting(pd.concat(dfs.values()).reset_index(drop=True), self.train_class_reweighting, f'{self.aux_var_prefix}eventWeightTrain')

        for filepath in filepaths:
            standardized_df = self.apply_standardization(dfs[filepath], fold)
            eos.save_file_eos(standardized_df, make_output_filepath(filepath[filepath.find(self.base_filepath):], self.output_dirpath, f"train{fold}"))

    def make_all_test(self, filepaths: list, force: bool=False):
        for fold in range(self.n_folds): self.make_test(filepaths, fold, force=force)
    def make_test(self, filepaths: list, fold: int, force: bool=False):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"

        for filepath in filepaths:
            df = self.make_df(filepath)
            mask = self.test_mask(df, fold)
            df = df.loc[mask].reset_index(drop=True)
            self.add_vars(df, fold, map_filepath_to_class(self.class_sample_map, filepath[filepath.find(self.base_filepath):]))
            
            standardized_df = self.apply_standardization(df, fold)
            eos.save_file_eos(standardized_df, make_output_filepath(filepath[filepath.find(self.base_filepath):], self.output_dirpath, f"test{fold}"), force=force)

    
    #############################################################
    # Retrieving
    def get_train(self, fold: int, syst_name: str='nominal', shuffle: bool=True):
        filepaths = get_traintest_filepaths_func(self.output_dirpath, dataset="train", syst_name=syst_name)(fold)

        df = pd.concat(
            [eos.load_file_eos(pd.DataFrame, filepath) for model_class in filepaths.keys() for filepath in filepaths[model_class]], 
            ignore_index=True
        )
        assert 'Data' not in set(np.unique(df[f'{self.aux_var_prefix}sample_name']).tolist()), f"Data is getting into train dataset... THIS IS VERY BAD"
        
        if shuffle:
            rng = np.random.default_rng(seed=self.seed)
            df.reindex(rng.permutation(df.index))

        return df
    
    def get_test(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        if regex == 'test_of_train':
            filepaths = get_traintest_filepaths_func(self.output_dirpath, dataset="test", syst_name=syst_name)(fold)
        else:
            filepaths = get_test_filepaths_func(self.output_dirpath, syst_name=syst_name, regex=regex)(fold)

        df = pd.concat(
            [eos.load_file_eos(pd.DataFrame, filepath) for model_class in filepaths.keys() for filepath in filepaths[model_class]], 
            ignore_index=True
        )

        return df
