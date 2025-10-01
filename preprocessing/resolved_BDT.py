# Stdlib packages
import argparse
import copy
import datetime
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
    f"{jet_prefix}_lead_bjet_bTagWPL", f"{jet_prefix}_lead_bjet_bTagWPM", f"{jet_prefix}_lead_bjet_bTagWPT",
    f"{jet_prefix}_lead_bjet_bTagWPXT", f"{jet_prefix}_lead_bjet_bTagWPXXT",
    # --------
    f'{jet_prefix}_sublead_bjet_eta', 
    # f"{jet_prefix}_sublead_bjet_btagPNetB",
    f"{jet_prefix}_sublead_bjet_bTagWPL", f"{jet_prefix}_sublead_bjet_bTagWPM", f"{jet_prefix}_sublead_bjet_bTagWPT",
    f"{jet_prefix}_sublead_bjet_bTagWPXT", f"{jet_prefix}_sublead_bjet_bTagWPXXT",
    
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
    'eventWeight',

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
MAKE_PLOTS = True

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "--input_filepaths", 
    default='',
    help="JSON for full filepaths (separated with \',\') on LPC"
)
parser.add_argument(
    "--output_dirpath", 
    default=CWD,
    help="Full filepath on LPC for output to be dumped"
)
parser.add_argument(
    "--force", 
    action="store_true",
    help="Flag to force the re-creation of the files"
)

################################


def plot_vars(df, output_dirpath, sample_name, title="pre-std, train"):
    std_type, df_type = tuple(title.split(", "))

    if "pre" in std_type: apply_logs(df)

    for var in df.columns:
        if log_standardize(var): var_label = f"ln({var})"
        else: var_label = var

        var_hist = hist.Hist(
            hist.axis.Regular(100, np.min(df[var]), np.max(df[var]), name="var", label=var_label), 
        ).fill(var=df[var])

        fig, ax = plt.subplot()
        hep.cms.lumitext(f"Run3" + r" (13.6 TeV)", ax=ax)
        hep.cms.text("Simulation", ax=ax)

        hep.histplot(var_hist, ax=ax, linewidth=3, histtype="step", yerr=True, label=" - ".join([sample_name, title]))
        plt.legend()

        plt.savefig(os.path.join(output_dirpath, "plots", f"{var}_{std_type.replace('-', '')}_{df_type}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dirpath, "plots", f"{var}_{std_type.replace('-', '')}_{df_type}.png"), bbox_inches='tight')
        plt.close()


def make_output_filepath(filepath, base_output_dirpath, extra_text):
    filename = filepath[filepath.rfind('/')+1:]
    output_dirpath = os.path.join(
        base_output_dirpath, 
        CURRENT_TIME, 
        filepath[filepath.find(BASE_FILEPATH):filepath.rfind('/')]
    )
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    filename = filename[:filename.rfind('.')] + f"_{extra_text}_{CURRENT_TIME}" + filename[filename.rfind('.'):]

    return os.path.join(output_dirpath, filename)

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
            df.loc[mask, col] = np.log(df[col])
    return df

def get_dfs(filepaths, BDT_vars, AUX_vars):
    dfs, aux_dfs = {}, {}
    for filepath in sorted(filepaths):
        dfs[filepath] = pq.read_table(filepath, columns=list(BDT_vars)).to_pandas()
        aux_dfs[filepath] = pq.read_table(filepath, columns=list(AUX_vars)).to_pandas()
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
    # Defining variables to use #
    if re.search('_EFT', output_dirpath) is None: 
        BDT_variables = BASIC_VARIABLES(JET_PREFIX) | MHH_CORRELATED_VARIABLES(JET_PREFIX)
    else:
        BDT_variables = BASIC_VARIABLES(JET_PREFIX)
    AUX_variables = AUX_VARIABLES(JET_PREFIX)
        
    train_dfs, train_aux_dfs = get_dfs(input_filepaths['train'], BDT_variables, AUX_variables)
    test_dfs, test_aux_dfs = get_dfs(input_filepaths['test'], BDT_variables, AUX_variables)

    for fold_idx in range(TRAIN_MOD):
        (
            train_dfs_fold, train_aux_dfs_fold, 
            test_dfs_fold, test_aux_dfs_fold 
        ) = get_split_dfs(input_filepaths['train-test'], BDT_variables, AUX_variables, fold_idx)

        x_mean, x_std = compute_standardization(train_dfs, train_dfs_fold)

        for filepath in train_dfs.keys():
            train_dfs_fold[filepath] = copy.deepcopy(train_dfs[filepath])
            train_aux_dfs_fold[filepath] = copy.deepcopy(train_aux_dfs[filepath])

        for filepath, df in train_dfs_fold.items():
            output_filepath = make_output_filepath(filepath, output_dirpath, f"train{fold_idx}")
            if MAKE_PLOTS: plot_vars(df, output_filepath, train_aux_dfs_fold[filepath]["AUX_sample_name"], title="pre-std, train")

            cols = list(df.columns)
            df = apply_logs(df)
            df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
            df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

            if MAKE_PLOTS: plot_vars(df, output_filepath, train_aux_dfs_fold[filepath]["AUX_sample_name"], title="post-std, train")

            for aux_col in train_aux_dfs_fold[filepath].columns:
                df[f"AUX_{aux_col}"] = train_aux_dfs_fold[filepath].loc[:,aux_col]

            df.to_parquet(output_filepath)


        for filepath in test_dfs.keys():
            test_dfs_fold[filepath] = copy.deepcopy(test_dfs[filepath])
            test_aux_dfs_fold[filepath] = copy.deepcopy(test_aux_dfs[filepath])

        for filepath, df in test_dfs_fold.items():
            output_filepath = make_output_filepath(filepath, output_dirpath, f"test{fold_idx}")
            if MAKE_PLOTS: plot_vars(df, output_filepath, test_aux_dfs_fold[filepath]["AUX_sample_name"], title="pre-std, test")

            cols = list(df.columns)
            df = apply_logs(df)
            df = (np.ma.array(df, mask=(df == FILL_VALUE)) - x_mean)/x_std
            df = pd.DataFrame(df.filled(FILL_VALUE), columns=cols)

            if MAKE_PLOTS: plot_vars(df, output_filepath, test_aux_dfs_fold[filepath]["AUX_sample_name"], title="post-std, test")

            for aux_col in test_aux_dfs_fold[filepath].columns:
                df[f"AUX_{aux_col}"] = test_aux_dfs_fold[filepath].loc[:,aux_col]

            df.to_parquet(output_filepath)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.input_filepaths, 'r') as f:
        input_filepaths = json.load(f)

    preprocess_resolved_bdt(input_filepaths, args.output_dirpath)
