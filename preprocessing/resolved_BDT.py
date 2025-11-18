# Stdlib packages
import argparse
import copy
import datetime
import glob
import json
import logging
import os
import re

# Common Py packages
import numpy as np  
import pandas as pd
import pyarrow.parquet as pq

# HEP packages
import hist
import mplhep as hep
import matplotlib.pyplot as plt

################################


from preprocessing_utils import match_sample
from retrieval_utils import argsorted

################################


# This map definition does not exclusively define the samples in the training/testing 
#  datasets, rather this map defines how to split the samples into distinct 
#  classes, as well as those classes' "true" labels as seen by the BDT. For example, 
#  in a training involving the SM ggF HH signal and *NOT* the other kl points, the
#  SM ggF HH signal sample might have a "true" label of 0, whereas the kl=5 
#  sample -- while still technically an HH "signal" -- is "unlabled" according
#  to the trained BDT because it didn't see the kl=5 sample during training.
#
# The structure of this map has the keys as the name of the classes, and the values
#  as a list of wildcard sample-names (i.e. glob formatting) of that class.
# {
#   'class 1': ['glob*name*1*', '*globname*2', etc]
# }
#
# The training dataset is defined by the intersection of this map *AND* the 
#  samples passed under the 'train' and 'train-test' keys of the input_flepaths JSON.
#  Therefore, if a sample is passed in the input JSON, but its glob-name is not 
#  entered here, the sample will not be included in the training. Similarly, if a
#  sample's glob-name is listed here, but the sample is not passed in the input JSON,
#  it will not be included in the training.
#
# Beyond the training files, this mapping is important because only the samples
#  listed in this map will be used for the baseline evaluation metrics (ROC curves,
#  Feature Importance, Confusion Matrix, etc). All samples in the train and test
#  datasets can be evaluated and used for plotting, but it is not the default behavior.
CLASS_SAMPLE_MAP = {
    'ggF HH': ["*GluGlu*HH*kl-1p00*"],  # *!batch
    'ttH + bbH': ["*ttH*", "*bbH*"],
    'VH': ["*VH*", "*ZH*", "*Wm*H*", "*Wp*H*"],
    'nonRes + ggFH + VBFH': ["*GGJets*", "*GJet*", "*TTGG*", "*GluGluH*GG*", "*VBFH*GG*"]
}
TRAIN_ONLY_SAMPLES = {
    "*Zto2Q*", "*Wto2Q*", "*batch[4-6]*"
}
TEST_ONLY_SAMPLES = {
    "*Data*", "*GluGlu*HH*kl-0p00*", "*GluGlu*HH*kl-2p45*", "*GluGlu*HH*kl-5p00*"
}

BASIC_VARIABLES = lambda jet_prefix: {
    # MET variables
    'puppiMET_pt',

    # jet vars
    f'{jet_prefix}_DeltaPhi_j1MET', f'{jet_prefix}_DeltaPhi_j2MET', 
    'n_jets', f'{jet_prefix}_chi_t0', f'{jet_prefix}_chi_t1',

    # lepton vars
    'lepton1_pt', 'lepton1_pfIsoId', 'lepton1_mvaID',
    'DeltaR_b1l1',

    # angular vars
    f'{jet_prefix}_CosThetaStar_CS', f'{jet_prefix}_CosThetaStar_jj', f'{jet_prefix}_CosThetaStar_gg',

    # bjet vars
    f'{jet_prefix}_lead_bjet_eta', # eta
    # f"{jet_prefix}_lead_bjet_btagPNetB",
    # f"{jet_prefix}lead_bjet_btagUParTAK4B",
    f"{jet_prefix}_lead_bjet_bTagWPL", f"{jet_prefix}_lead_bjet_bTagWPM", f"{jet_prefix}_lead_bjet_bTagWPT",
    f"{jet_prefix}_lead_bjet_bTagWPXT", f"{jet_prefix}_lead_bjet_bTagWPXXT",
    # f"{jet_prefix}_lead_bjet_bTagWP3XT", f"{jet_prefix}_lead_bjet_bTagWP4XT",
    # --------
    f'{jet_prefix}_sublead_bjet_eta', 
    # f"{jet_prefix}_sublead_bjet_btagPNetB",
    # f"{jet_prefix}sublead_bjet_btagUParTAK4B",
    f"{jet_prefix}_sublead_bjet_bTagWPL", f"{jet_prefix}_sublead_bjet_bTagWPM", f"{jet_prefix}_sublead_bjet_bTagWPT",
    f"{jet_prefix}_sublead_bjet_bTagWPXT", f"{jet_prefix}_sublead_bjet_bTagWPXXT",
    # f"{jet_prefix}_sublead_bjet_bTagWP3XT", f"{jet_prefix}_sublead_bjet_bTagWP4XT",
    
    # diphoton vars
    'eta',

    # Photon vars
    'lead_mvaID', 'lead_sigmaE_over_E',
    # --------
    'sublead_mvaID', 'sublead_sigmaE_over_E',
    
    # HH vars
    f'{jet_prefix}_HHbbggCandidate_pt', 
    f'{jet_prefix}_HHbbggCandidate_eta', 
    f'{jet_prefix}_pt_balance',

    # ZH vars
    f'{jet_prefix}_DeltaPhi_jj',
    f'{jet_prefix}_DeltaPhi_isr_jet_z',
}
MHH_CORRELATED_VARIABLES = lambda jet_prefix: {
    # MHH
    # f'{jet_prefix}_HHbbggCandidate_mass',

    # MET variables
    'puppiMET_sumEt',  #eft

    # jet vars
    f'{jet_prefix}_DeltaR_jg_min',  #eft

    # dijet vars
    f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),  #eft
    f'{jet_prefix}_dijet_pt',  #eft

    # bjet vars
    f'{jet_prefix}_lead_bjet_pt', #eft
    f'{jet_prefix}_lead_bjet_sigmapT_over_pT', #eft
    f'{jet_prefix}_lead_bjet_pt_over_Mjj', #eft
    # --------
    f'{jet_prefix}_sublead_bjet_pt', #eft
    f'{jet_prefix}_sublead_bjet_sigmapT_over_pT', #eft
    f'{jet_prefix}_sublead_bjet_pt_over_Mjj', #eft

    # diphoton vars
    'pt',  #eft

    # ZH vars
    f'{jet_prefix}_DeltaEta_jj', #eft
    f'{jet_prefix}_isr_jet_pt',  #eft
}
AUX_VARIABLES = lambda jet_prefix: {
    # identifiable event info
    'event', 'lumi', 'hash', 'sample_name', 

    # MC info
    'weight', 'eventWeight',

    # mass
    'mass', 
    f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),
    f'{jet_prefix}_HHbbggCandidate_mass',

    # sculpting study
    f'{jet_prefix}_max_nonbjet_btag',

    # event masks
    f'{jet_prefix}_resolved_BDT_mask',
}

FILL_VALUE = -999
TRAIN_MOD = 5
JET_PREFIX = 'nonRes'

SEED = 21
BASE_FILEPATH = 'Run3_202'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
CWD = os.getcwd()
DF_MASK = 'default'  # 'none'

DEBUG = False
DRYRUN = False
MAKE_PLOTS = False
REMAKE_TEST = False

END_FILEPATH = "preprocessed.parquet"

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "--input_eras", 
    default='',
    help="JSON for filepaths on cluster for eras"
)
parser.add_argument(
    "--output_dirpath", 
    default=CWD,
    help="Full filepath on LPC for output to be dumped"
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

################################


def get_input_filepaths(input_eras):
    input_filepaths = {'train-test': list(), 'train': list(), 'test': list()}
    with open(input_eras, 'r') as f:
        for line in f:
            stdline = line.strip()
            if stdline[0] == "#": continue

            sample_filepaths = glob.glob(os.path.join(stdline, "**", f"*{END_FILEPATH}"), recursive=True)
            for sample_filepath in sample_filepaths:
                if match_sample(sample_filepath, TEST_ONLY_SAMPLES) is not None:
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
    if DEBUG:
        print(input_filepaths)
    return input_filepaths

def plot_vars(df, output_dirpath, sample_name, title="pre-std, train0"):
    std_type, df_type = tuple(title.split(", "))
    plot_dirpath = os.path.join(output_dirpath, "plots", "_".join([std_type.replace('-', ''), df_type]))
    if not os.path.exists(plot_dirpath): os.makedirs(plot_dirpath)

    if "pre" in std_type: apply_logs(df)

    if DEBUG:
        print('='*60+'\n'+'='*60)
        print(output_dirpath)

    for var in df.columns:
        if log_standardize(var): var_label = f"ln({var})"
        else: var_label = var

        var_mask = (
            (df[var] != FILL_VALUE)
            & np.isfinite(df[var])
        )
        good_var_bool = np.any(var_mask) and np.min(df.loc[var_mask, var]) != np.max(df.loc[var_mask, var])

        max_val = np.max(df.loc[var_mask, var]) if good_var_bool else 0.
        min_val = np.min(df.loc[var_mask, var]) if good_var_bool else 1.

        if DEBUG:
            print('-'*60)
            print(var)
            print(f"min = {min_val}, max = {max_val}")

        var_hist = hist.Hist(
            hist.axis.Regular(100, min_val, max_val, name="var", label=var_label, growth=True), 
        ).fill(var=df.loc[var_mask, var])

        fig, ax = plt.subplots()
        hep.cms.lumitext(f"Run3" + r" (13.6 TeV)", ax=ax)
        hep.cms.text("Simulation", ax=ax)

        hep.histplot(var_hist, ax=ax, histtype="step", yerr=True, label=" - ".join([sample_name, title]))
        plt.legend()
        plt.yscale('log')

        plt.savefig(os.path.join(plot_dirpath, f"{var}_{std_type.replace('-', '')}_{df_type}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_dirpath, f"{var}_{std_type.replace('-', '')}_{df_type}.png"), bbox_inches='tight')
        plt.close()


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

def no_standardize(column):
    no_std_terms = {
        'phi', 'eta',  # angular
        'id',  # IDs
        'btag'  # bTags
    }
    return any(no_std_term in column.lower() for no_std_term in no_std_terms)

def log_standardize(column):
    log_std_terms = {
        'pt', 'chi',
    }
    return any(log_std_term in column for log_std_term in log_std_terms)

def apply_logs(df):
    for col in df.columns:
        if log_standardize(col):
            mask = (df[col].to_numpy() > 0)
            df.loc[mask, col] = np.log(df.loc[mask, col])
    return df

def get_dfs(filepaths, BDT_vars, AUX_vars):
    dfs, aux_dfs = {}, {}
    for filepath in sorted(filepaths):
        pq_file = pq.ParquetFile(filepath)
        for pq_batch in pq_file.iter_batches(batch_size=524_288, columns=BDT_vars+AUX_vars):
            df_batch = pq_batch.to_pandas()
            df_mask = get_df_mask(df_batch)
            dfs[filepath] = df_batch.loc[df_mask, BDT_vars].reset_index(drop=True)
            aux_dfs[filepath] = df_batch.loc[df_mask, AUX_vars].reset_index(drop=True)

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
    if not DRYRUN:
        class_sample_map_filepath = os.path.join(output_dirpath, 'class_sample_map.json')
        with open(class_sample_map_filepath, 'w') as f:
            json.dump(CLASS_SAMPLE_MAP, f)

    # Defining variables to use #
    if re.search('_EFT', output_dirpath) is None: 
        BDT_variables = BASIC_VARIABLES(JET_PREFIX) | MHH_CORRELATED_VARIABLES(JET_PREFIX)
    else:
        BDT_variables = BASIC_VARIABLES(JET_PREFIX)
    AUX_variables = AUX_VARIABLES(JET_PREFIX)
    BDT_variables, AUX_variables = sorted(BDT_variables), sorted(AUX_variables)
    
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
    args = parser.parse_args()

    print('='*60)
    print(f'Starting Resolved BDT processing at {CURRENT_TIME}')

    DEBUG = args.debug
    DRYRUN = args.dryrun
    MAKE_PLOTS = args.plots
    REMAKE_TEST = args.remake_test
    args_output_dirpath = os.path.normpath(args.output_dirpath)
    if REMAKE_TEST: CURRENT_TIME = args_output_dirpath[args_output_dirpath.rfind('/')+1:]
    else: args_output_dirpath = os.path.join(args_output_dirpath, CURRENT_TIME)
    if not os.path.exists(args_output_dirpath) and not DRYRUN:
        os.makedirs(args_output_dirpath)
    input_filepaths = get_input_filepaths(args.input_eras)

    preprocess_resolved_bdt(input_filepaths, args_output_dirpath)
    print(f'Finished Resolved BDT processing')
