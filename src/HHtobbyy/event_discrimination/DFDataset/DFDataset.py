# Stdlib packages
import datetime
import json
import operator
import os
from collections.abc import Callable

# Common Py packages
import numpy as np  
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

# ML packages
from sklearn.model_selection import train_test_split

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset.DFDataset_utils import (
    map_filepath_to_class, make_output_filepath,
    identity, logzscore, 
    equalProc_train_test_split, # random_oversample, random_undersample
)
from HHtobbyy.workspace_utils.retrieval_utils import (
    FILL_VALUE, match_sample, match_regex, 
    multifold, sub_filepath, 
    batched_writer, batched_executor
)


class DFDataset:
    """
    Defines the pandas DataFrame format for the dataset. DataFrame is useful as a common
    format that can easily be converted downstream to the model-specific format (see ModelDataset).
    """

    def __init__(self, config: str|dict):
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

        if type(config) is str: 
            if config.endswith('.json'):
                eos_filepath = eos.load_file_eos(config)
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            elif config.split('/')[-1].find('.') < 0:
                print(f"WARNING: Config directory supplied rather than file, attempting to load with default filename... ")
                eos_filepath = eos.load_file_eos(os.path.join(config, self.config_filename))
                with open(eos_filepath, 'r') as f: config = json.load(f)
                eos.delete_lockfile(eos_filepath)
            else:
                raise IOError(f"ERROR: Config file does not appear to be a json, only JSON is supported currently. ")
            

        #########################
        # REQUIRED CONFIG KEYS
        # Dirpath to dump dataset
        self.output_dirpath = ''

        # Short tagline to describe dataset
        self.dataset_tag = ''

        # Model variables used in training
        self.model_vars = []

        # Auxiliary variables not used in training, but important downstream
        self.aux_vars = []

        # Mapping between sample filenames and class groupings
        self.class_sample_map = {}
        #########################

        # Boolean to sort the input variables alphabetically
        self.sort_inputs = True

        # RNG seed
        self.seed = 21

        # Fill value for bad data from preprocessing
        self.fill_value = FILL_VALUE

        # Fill value for bad data to go into DFDataset
        self.refill_value = FILL_VALUE

        # Mask variable for preselection, None for no extra pre-selection
        self.presel_filter = None

        # Number of folds, one model per fold
        self.n_folds = 5

        # Fraction of training data to use for validation, if using
        self.val_split = 0.2

        # Method to split train and val
        self.train_val_split_method = 'scikit'

        # Method used for the standardization
        self.standardization_method = 'logzscore'

        # Standard prefix for auxiliary variables, i.e. variables useful 
        #   for event-identification but *not* used in the training
        self.aux_var_prefix = 'AUX_'

        # Variable to use when matching up two DFs (with the same events) to combine correctly
        self.sort_var = "hash"
        # Variable to use for genweight / sum_of_genw normalization
        self.gen_weight_var = "weight"
        # Variable to use for genweight * xs * lumi / sum_of_genw normalization
        self.event_weight_var = "eventWeight"
        # Variable to use for separating by sample
        self.sample_var = "sample_name"
        # Variable to use for separating by era
        self.era_var = "sample_era"

        # End of filepath for files to pull using eras selection
        #  (doesn't matter if passing filepaths directly)
        self.filepostfix = 'preprocessed.parquet'

        # Basic fileprefix to separate local machine directories from HiggsDNA 
        #   (or equivalent preprocessor) directories
        self.base_filepath = 'Run[1-3]_20'

        # Optional methods to undersample or oversample train dataset
        self.overundersample_train_per_proc = 'none'

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

        if "sort_inputs" in config: self.sort_inputs = config["sort_inputs"]
        for key, value in config.items():
            if hasattr(self, key): 
                setattr(self, key, sorted(value) if type(value) is list and self.sort_inputs else value)

        # All variables
        self.all_vars_map = {
            model_var: model_var for model_var in self.model_vars
        } | {
            self.aux_var_prefix + aux_var: aux_var for aux_var in self.aux_vars
        }

        # Number of classes
        self.n_classes = len(self.class_sample_map)

        # Checking if config already exists at output_dirpath
        #   otherwise creating dirpath and saving
        if not self.check_output_dirpath():
            self.dataset_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'
            self.output_dirpath = os.path.join(self.output_dirpath, f"{self.dataset_tag}_{self.dataset_time}")

            config_filepath = os.path.join(self.output_dirpath, self.config_filename)
            eos_filepath = eos.save_file_eos(config_filepath)
            with open(eos_filepath, 'w') as f: json.dump(self.__dict__, f)
            eos.delete_lockfile(eos_filepath)

        # process pc variables for fast pyarrow dataset loading
        self.process_presel_filter()
        self.process_var_map()

        # lambda func to map filepath to class idx for readability
        self.class_idx = lambda filepath: map_filepath_to_class(self.class_sample_map, sub_filepath(filepath, self.base_filepath))
        self.out_filepath = lambda in_filepath, fold, dataset: make_output_filepath(sub_filepath(in_filepath, self.base_filepath), self.output_dirpath, f"{dataset}{fold}")

    def check_output_dirpath(self):
        config_filepath = os.path.join(self.output_dirpath, self.config_filename)
        return eos.file_exists_eos(config_filepath)
    
    def process_presel_filter(self):
        """
        Processes input presel_filter in config (needs to be JSON serializable) to pyarrow.dataset.Expression format.
        Expected format is list[list[tuple]], detailed below:
         - tuple[str, str, float]: (column_name, logical op, cut_value)
         - list[tuple]: logical-and of the tuple cuts
         - list[list]: logical-or of complex and-ed cuts
        """
        ops = {
            '<': operator.lt, '<=': operator.le, '==': operator.eq, '>=': operator.ge, '>': operator.gt
        }
        if self.presel_filter is None: return
        else:
            assert type(self.presel_filter) is list, f"Input presel_filter needs to be of type list[list[tuple]]"
            ored_filter = None
            for or_list in self.presel_filter:
                assert type(or_list) is list, f"Input presel_filter needs to be of type list[list[tuple]]"
                anded_filter = None
                for and_tuple in or_list:
                    assert type(and_tuple) is tuple, f"Input presel_filter needs to be of type list[list[tuple]]"
                    exp = ops[and_tuple[1]](pc.field(and_tuple[0]), and_tuple[2])
                    if anded_filter is None: anded_filter = exp
                    else: anded_filter = (anded_filter & exp)
                if ored_filter is None: ored_filter = anded_filter
                else: ored_filter = (ored_filter | anded_filter)
            self.presel_filter = ored_filter

    def process_var_map(self):
        """
        Processes all_vars_map for passing to dataset as columns.
        Expected format is dict[str: str], detailed below:
         - key[str]: final column name
         - value[str]: initial column name
        """
        pc_all_vars = {}
        for final_col, init_col in self.all_vars_map.items():
            pc_all_vars[final_col] = pc.field(init_col)
        self.all_vars_map = pc_all_vars

    #############################################################
    # Event masking
    def train_mask(self, df: pd.DataFrame, fold: int, accumulation: dict={}, **kwargs):
        if self.n_folds > 1: mask = np.asarray(df[f'{self.aux_var_prefix}event'].mod(self.n_folds).ne(fold), dtype=bool)
        else: mask = np.ones(len(df), dtype=bool)
        return np.logical_and(mask, self.over_under_sample(df, accumulation))
    def test_mask(self, df: pd.DataFrame, fold: int, **kwargs):
        return np.asarray(df[f'{self.aux_var_prefix}event'].mod(self.n_folds).eq(fold), dtype=bool)
    

    #############################################################
    # Basic DF build
    def make_df(self, df: pd.DataFrame, fold: int, class_idx: int, mask_func, accumulation: dict={}, **kwargs):
        mask = mask_func(df, fold, accumulation=accumulation, **kwargs)
        df = df.loc[mask].reset_index(drop=True)
        self.add_vars(df, class_idx, accumulation)
        self.apply_standardization(df, fold)
        self.good_df(df)
        return df
    
    def accumulate_dataset(self, df: pd.DataFrame, fold: int, accumulation: dict, **kwargs):
        mask = self.train_mask(df, fold)
        df = df.loc[mask].reset_index(drop=True)

        # accumulate E[X], E[X^2], and N for z-score-like standardization
        for model_var in self.model_vars:
            if self.standardization_method+model_var not in accumulation.keys(): 
                accumulation[self.standardization_method+model_var] = {'exp_x': 0., 'exp_xsq': 0., 'N': 0}
            masked_col = globals(self.standardization_method)(model_var, np.ma.array(df[model_var], mask=(df[model_var] == self.fill_value)))
            df_accumulation_col = {'exp_x': masked_col.sum(), 'exp_xsq': np.ma.power(masked_col).sum(), 'N': masked_col.count()}
            accumulation[self.standardization_method+model_var] = {
                key: sum(pair) for key, pair in zip(
                    accumulation[self.standardization_method+model_var].keys(), 
                    zip(accumulation[self.standardization_method+model_var].values(), df_accumulation_col.values())
                )
            }

        # accumulate sum of processes for re-sampling
        df_proc = df[f'{self.aux_var_prefix}{self.sample_var}'][0]
        if self.event_weight_var+df_proc not in accumulation.keys(): 
            accumulation[self.event_weight_var+df_proc] = {'sum': 0., 'N': 0}
        masked_weight = np.ma.array(df[self.event_weight_var], mask=(df[self.event_weight_var] == self.fill_value))
        df_accumulation_class = {'sum': masked_weight.sum(), 'N': masked_weight.count()}
        accumulation[self.event_weight_var+df_proc] = {
            key: sum(pair) for key, pair in zip(
                accumulation[self.event_weight_var+df_proc].keys(), 
                zip(accumulation[self.event_weight_var+df_proc].values(), df_accumulation_class.values())
            )
        }

        # accumulate sum of classes for class-standardization
        df_classTag = list(self.class_sample_map.keys())[df[f'{self.aux_var_prefix}label1D'][0]]
        if self.event_weight_var+df_classTag not in accumulation.keys(): 
            accumulation[self.event_weight_var+df_classTag] = {'sum': 0., 'N': 0}
        masked_weight = np.ma.array(df[self.event_weight_var], mask=(df[self.event_weight_var] == self.fill_value))
        df_accumulation_class = {'sum': masked_weight.sum(), 'N': masked_weight.count()}
        accumulation[self.event_weight_var+df_classTag] = {
            key: sum(pair) for key, pair in zip(
                accumulation[self.event_weight_var+df_classTag].keys(), 
                zip(accumulation[self.event_weight_var+df_classTag].values(), df_accumulation_class.values())
            )
        }
        
    def good_df(self, df: pd.DataFrame):
        assert not df.isnull().values.any(), f"ERROR: DFDataset contains NaN values, something likely went wrong with the DF mergings"
        assert set(self.all_vars_map.keys()).issubset(set(df.columns)), f"ERROR: DFDataset missing necessary columns: {set(self.all_vars_map.keys()) - set(df.columns)}"

    #############################################################
    # Additional variables
    def sample_reweighting(self, df: pd.DataFrame, sample_reweight: str|dict, reweight_var: str):
        if type(sample_reweight) is str and sample_reweight == 'none': return
        else:
            df_era, df_proc = df[f'{self.aux_var_prefix}{self.era_var}'][0], df[f'{self.aux_var_prefix}{self.sample_var}'][0]
            for era_proc, reweight in sample_reweight.items():
                era, proc = tuple(era_proc.split('<>'))
                if not ((df_era == era or era == 'all') and df_proc == proc): continue
                df[reweight_var] = df[reweight_var] * reweight

    def class_reweighting(self, df: pd.DataFrame, class_reweight: str|dict, reweight_var: str, accumulation: dict):
        if type(class_reweight) is str and class_reweight == 'none': return
        else:
            df_classIdx = df[f'{self.aux_var_prefix}label1D'][0]
            for classTag, (reweight, ref_classTags) in class_reweight.items():
                classTagIdx = list(self.class_sample_map.keys()).index(classTag)
                if df_classIdx != classTagIdx: continue
                classSum, ref_classSum = accumulation[self.event_weight_var+classTag]['sum'], sum([accumulation[self.event_weight_var+ref_classTag]['sum'] for ref_classTag in ref_classTags])
                df[reweight_var] = df[reweight_var] * reweight * ref_classSum / classSum

    def add_vars(self, df: pd.DataFrame, class_idx: int, accumulation: dict):
        df[f'{self.aux_var_prefix}label1D'] = class_idx

        # Reweighting eventWeight if improperly preprocessed
        self.sample_reweighting(df, self.test_sample_reweighting, f'{self.aux_var_prefix}{self.event_weight_var}')

        # Creating and reweighting eventWeightTrain for training
        df[f'{self.aux_var_prefix}{self.event_weight_var}Train'] = df[f'{self.aux_var_prefix}{self.event_weight_var}']
        self.class_reweighting(df, self.train_class_reweighting, f'{self.aux_var_prefix}{self.event_weight_var}Train', accumulation)
        self.sample_reweighting(df, self.train_sample_reweighting, f'{self.aux_var_prefix}{self.event_weight_var}Train')


    #############################################################
    # Standardization
    def compute_standardization(self, fold: int, accumulation: dict):
        stddict = {'col': [], 'mean': [], 'std': []}

        for key, accums in accumulation.items():
            if not key.startswith(self.standardization_method): continue
            stddict['col'].append(key.replace(self.standardization_method, ''))
            stddict['mean'].append(accums['exp_x'] / accums['N'])
            stddict['std'].append(((accums['exp_xsq'] / accums['N']) - (accums['exp_x'] / accums['N'])**2)**0.5)

        stddict_filepath = os.path.join(self.output_dirpath, f'{self.standardization_method.lower()}_{self.standardization_subfilename}{fold}.json')
        eos_filepath = eos.save_file_eos(stddict_filepath)
        with open(eos_filepath, 'w') as f: json.dump(stddict, f)
        eos.delete_lockfile(eos_filepath)

    def apply_standardization(self, df: pd.DataFrame, fold: int):
        stddict_filepath = os.path.join(self.output_dirpath, f'{self.standardization_method.lower()}_{self.standardization_subfilename}{fold}.json')
        eos_filepath = eos.load_file_eos(stddict_filepath)
        with open(eos_filepath, 'r') as f: stddict = json.load(f)
        eos.delete_lockfile(eos_filepath)

        for model_var, mean, std in zip(stddict['col'], stddict['mean'], stddict['std']):
            masked_col = globals()[self.standardization_method](model_var, np.ma.array(df[model_var], mask=(df[model_var] == self.fill_value)))
            masked_col = (masked_col - mean) / std
            df[model_var] = masked_col.filled(self.refill_value)

    #############################################################
    # Oversample/Undersample for training
    def over_under_sample(self, df: pd.DataFrame, accumulation: dict):
        if self.overundersample_train_per_proc == 'none' and self.undersample_train_per_proc == 'none': return
        else: globals(self.overundersample_train_per_proc)(df, accumulation, self.seed)

    #############################################################
    # Train/Val splitting
    def train_val_split(self):
        if self.train_val_split_method == 'scikit': return train_test_split
        elif self.train_val_split_method == 'equalProc': return equalProc_train_test_split
        else: raise NotImplementedError(f"Train/Val split method not yet implemented, use \'scikit\'.")


    #############################################################
    # Building
    def make_all_train(self, filepaths: list, **kwargs):
        multifold(self.make_train, (filepaths, ), self.n_folds, **kwargs)
    def make_train(self, fold: int, filepaths: list, force: bool=False, **kwargs):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"
        if not force: filepaths = self.get_new_filepaths(fold, filepaths, 'train')

        accumulation = {}
        for filepath in filepaths:
            batched_executor(self.get_df_iter, filepath)(self.accumulate_dataset)(
                fold, accumulation, columns=self.all_vars_map, filter=self.presel_filter, **kwargs
            )
        self.compute_standardization(fold, accumulation)

        for filepath in filepaths:
            batched_writer(self.get_df_iter, filepath, self.out_filepath(filepath, fold, 'train'))(self.make_df)(
                fold, self.class_idx(filepath), self.train_mask, accumulation=accumulation,
                columns=self.all_vars_map, filter=self.presel_filter, **kwargs
            )
            

    def make_all_test(self, filepaths: list, **kwargs):
        multifold(self.make_test, (filepaths, ), self.n_folds,  **kwargs)
    def make_test(self, fold: int, filepaths: list, force: bool=False, **kwargs):
        assert fold >= 0 and fold < self.n_folds, f"ERROR: Expected a fold index between 0 and {self.n_folds}, received {fold}"
        if not force: filepaths = self.get_new_filepaths(fold, filepaths, 'test')
        
        for filepath in filepaths:
            print(f'Making - {filepath}')
            batched_writer(self.get_df_iter, filepath, self.out_filepath(filepath, fold, 'test'))(self.make_df)(
                fold, self.class_idx(filepath), self.test_mask, 
                columns=self.all_vars_map, filter=self.presel_filter, **kwargs
            )

    
    #############################################################
    # Retrieving
    def get_df_iter(self, filepath: str, batch_size: int=131_072, columns: None|list[str]=None, filter: None|ds.Expression=None, **kwargs):
        dataset = ds.dataset(filepath, format="parquet")
        return dataset.to_batches(batch_size=batch_size, columns=columns, filter=filter)
    
    def get_all_train(self, syst_name: str='nominal', shuffle: bool=True, **kwargs):
        dfs = []
        for fold in range(self.n_folds): dfs.append(self.get_train(fold, syst_name=syst_name, shuffle=shuffle, **kwargs))
        return pd.concat(dfs, ignore_index=True)
    def get_train(self, fold: int, syst_name: str='nominal', shuffle: bool=True, **kwargs):
        filepaths = self.get_traintest_filepaths(fold, dataset="train", syst_name=syst_name)

        df = pd.concat(
            [self.get_df(filepath, **kwargs) for model_class in filepaths.keys() for filepath in filepaths[model_class]], 
            ignore_index=True, join="inner"
        )
        
        assert 'Data' not in set(df[f'{self.aux_var_prefix}{self.sample_var}'].unique().tolist()), f"Data is getting into train dataset... THIS IS VERY BAD"
        
        if shuffle:
            rng = np.random.default_rng(seed=self.seed)
            df.reindex(rng.permutation(df.index))

        return df
    
    def get_all_test(self, syst_name: str='nominal', regex: str|list[str]='test_of_train', **kwargs):
        dfs = []
        for fold in range(self.n_folds): dfs.append(self.get_test(fold, syst_name=syst_name, regex=regex, **kwargs))
        return pd.concat(dfs, ignore_index=True)
    def get_test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train', **kwargs):
        if regex == 'test_of_train':
            filepaths = self.get_traintest_filepaths(fold, dataset="test", syst_name=syst_name)
        else:
            filepaths = self.get_test_filepaths(fold, syst_name=syst_name, regex=regex)

        df = pd.concat(
            [self.get_df(filepath, **kwargs) for model_class in filepaths.keys() for filepath in filepaths[model_class]], 
            ignore_index=True
        )

        return df
    

    #############################################################
    # Save new version of DF
    def sort_dfs(self, sorter: pd.DataFrame, sortee: pd.DataFrame):
        sortee_order = sortee[f"{self.aux_var_prefix}{self.sort_var}"].argsort()
        sorter_order = sorter[f"{self.aux_var_prefix}{self.sort_var}"].argsort()
        sortee_reorder = np.argsort(sorter_order)

        return sortee_order[sortee_reorder]
            
    
    #############################################################
    # Retrieve train/test files
    def get_new_filepaths(self, fold: int, input_filepaths: list[str], dataset: str, **kwargs):
        all_filepaths = set([filepath for filepath_list in self.get_traintest_filepaths(fold, dataset=dataset, syst_name='').values() for filepath in filepath_list])
        if dataset == 'test':
            all_filepaths = all_filepaths | set([filepath for filepath_list in self.get_test_filepaths(fold, syst_name='').values() for filepath in filepath_list])
        output_filepaths = {self.out_filepath(filepath, fold, dataset): filepath for filepath in input_filepaths}
        new_filepaths = sorted([output_filepaths[key] for key in list(set(output_filepaths.keys()) - all_filepaths)])
        return new_filepaths
    def get_traintest_filepaths(self, fold: int, dataset: str="train", syst_name: str='nominal', **kwargs):
        return {
            class_name: sorted(
                set(
                    sample_filepath
                    for sample_filepath in eos.glob_eos(os.path.join(self.output_dirpath, "**", f"*{dataset}{fold}.parquet"), recursive=True)
                    if (
                        (syst_name == "nominal" and match_sample(sample_filepath[len(self.output_dirpath):], ["_up", "_down"]) is None) 
                        or match_sample(sample_filepath[len(self.output_dirpath):], [syst_name]) is not None
                    ) and match_sample(sample_filepath[len(self.output_dirpath):], sample_names) is not None
                )
            ) for class_name, sample_names in self.class_sample_map.items()
        }
    def get_test_filepaths(self, fold: int, syst_name: str='nominal', regex: str|list[str]='', **kwargs):
        return {
            'test': sorted(
                set(
                    sample_filepath
                    for sample_filepath in eos.glob_eos(os.path.join(self.output_dirpath, "**", f"*test{fold}.parquet"), recursive=True)
                    if ( 
                        (syst_name == "nominal" and match_sample(sample_filepath[len(self.output_dirpath):], ["_up", "_down"]) is None) 
                        or match_sample(sample_filepath[len(self.output_dirpath):], [syst_name]) is not None
                    ) and match_sample(sample_filepath[len(self.output_dirpath):], [regex] if type(regex) is str else regex) is not None
                )
            )
        }
