# Stdlib packages
import argparse
import copy
import json
import logging
import os
import subprocess
import sys

# Common Py packages
import numpy as np
import pandas as pd
import prettytable as pt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# HEP packages
# import mplhep as hep
import hist

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
sys.path.append(os.path.join(GIT_REPO, "training/"))
sys.path.append(os.path.join(GIT_REPO, "evaluation/"))

from preprocessing_utils import match_regex, match_sample
from retrieval_utils import (
    get_class_sample_map, get_n_folds,
    get_train_Dataframe, get_test_subset_Dataframes
)
from training_utils import (
    get_dataset_dirpath,
)
from evaluation_utils import (
    transform_preds_options, transform_preds_func
)
import categorization_utils

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath", 
    help="Full filepath on LPC for trained model files"
)
parser.add_argument(
    "options_filepath",
    help="Full filepath on LPC for categorization options file"
)
parser.add_argument(
    "--dataset_dirpath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_dirpath.txt` file"
)
parser.add_argument(
    "--dataset", 
    choices=["train", "train-test"], 
    default="train-test",
    help="Evaluate and save out evaluation for what dataset"
)
parser.add_argument(
    "--minimal_dataset",
    action="store_true",
    help="Evaluate only for small dataset (useful for debugging)"
)
parser.add_argument(
    "--opt_sideband",
    choices=["none", "data", "mc"],
    help="Performs category optimization using fit to sidebands for estimation of non-resonant background in SR"
)
parser.add_argument(
    "--syst_name", 
    choices=["nominal", "all"], 
    default="nominal",
    help="Evaluate and save out evaluation for what systematic of a dataset"
)
parser.add_argument(
    "--discriminator", 
    choices=transform_preds_options(),
    default=transform_preds_options()[0],
    help="Defines the discriminator to use for categorization, discriminators are implemented in evaluation_utils"
)
parser.add_argument(
    "--signal",
    type=int,
    default=0,
    help="If the 1D label to use as signal during the optimization is not 0 (i.e. signal class is not first entry in class_sample_map), use this option to change"
)
parser.add_argument(
    "--verbose",
    action='store_true',
    help="Prints out category information as they're created"
)

################################


args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
if args.dataset_dirpath is None:
    DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
else:
    DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')
DATASET = args.dataset
SYST_NAME = args.syst_name
DISCRIMINATOR = args.discriminator
SIGNAL_LABEL = args.signal
VERBOSE = args.verbose
MINIMAL = args.minimal_dataset
OPT_SIDEBAND = args.opt_sideband
OPTIONS_FILEPATH = args.options_filepath

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

CATEGORIZATION_DIRPATH = os.path.join(TRAINING_DIRPATH, "categorization", "")
if not os.path.exists(CATEGORIZATION_DIRPATH): 
    os.makedirs(CATEGORIZATION_DIRPATH)
CATEGORIZATION_FILEPATH = os.path.join(CATEGORIZATION_DIRPATH, f"{DISCRIMINATOR}{f'_sig{CLASS_NAMES[SIGNAL_LABEL]}' if SIGNAL_LABEL != 0 else ''}_categorization.json")
CATEGORIZATION_OPTIONS_FILEPATH = os.path.join(CATEGORIZATION_DIRPATH, f"{DISCRIMINATOR}{f'_sig{CLASS_NAMES[SIGNAL_LABEL]}' if SIGNAL_LABEL != 0 else ''}_categorization_options.json")
with open(OPTIONS_FILEPATH, 'r') as f: CATEGORIZATION_OPTIONS = json.load(f)

TRANSFORM_LABELS, TRANSFORM_PREDS, TRANSFORM_CUT = transform_preds_func(CLASS_NAMES, DISCRIMINATOR, cutdirbool=True)
TRANSFORM_COLUMNS = [f"AUX_{transform_label}_prob" for transform_label in TRANSFORM_LABELS]
assert len(TRANSFORM_COLUMNS) <= 4, f"You're trying to run categorization over a discriminator with more than 4 dimensions, this likely won't converge with the brute-force way in this file. Please write your own categorization code or use a different discriminator"
CATEGORIZATION_COLUMNS = ['AUX_event', 'AUX_sample_name', 'AUX_eventWeight', 'AUX_mass', 'AUX_*_resolved_BDT_mask', 'AUX_label1D'] + TRANSFORM_COLUMNS

CATEGORIZATION_OPTIONS['TRANSFORM_COLUMNS'] = TRANSFORM_COLUMNS
CATEGORIZATION_OPTIONS['SIGNAL_LABEL'] = SIGNAL_LABEL
CATEGORIZATION_METHOD = getattr(categorization_utils, CATEGORIZATION_OPTIONS['METHOD'])
if 'STARTSTOPS' not in CATEGORIZATION_OPTIONS.keys():
    CATEGORIZATION_OPTIONS['STARTSTOPS'] = [[0., 1.] if '<' in TRANSFORM_CUT[i] else [1., 0.] for i in range(len(TRANSFORM_COLUMNS))]

NONRES_MC_SAMPLENAMES = {'TTGG', 'GJet', 'GGJets', 'DDQCDGJets', 'SherpaNLO'}

################################


def categorize_model():
    categories_dict = {}
    table = pt.PrettyTable()

    full_df_eval = None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        MC_df, MC_aux = get_train_Dataframe(DATASET_DIRPATH, fold_idx, dataset='test' if DATASET == 'train-test' else 'train', minimal=MINIMAL)
        for col in MC_aux.columns: MC_df.loc[:, col] = MC_aux.loc[:, col]
        cat_cols = [match_regex(cat_col, MC_df.columns) for cat_col in CATEGORIZATION_COLUMNS]
        assert set(cat_cols) <= set(MC_df.columns), f"The requested list of columns are not all in the file. Missing variables:\n{set(cat_cols) - set(MC_df.columns)}"
        MC_eval = MC_df.loc[:, cat_cols]

        SHERPA_df, SHERPA_aux = get_test_subset_Dataframes(DATASET_DIRPATH, fold_idx, ['SherpaNLO'], minimal=MINIMAL)
        for col in SHERPA_aux.columns: SHERPA_df.loc[:, col] = SHERPA_aux.loc[:, col]
        assert set(cat_cols) <= set(SHERPA_df.columns), f"The requested list of columns are not all in the Sherpa file. Missing variables:\n{set(cat_cols) - set(SHERPA_df.columns)}"
        MC_eval = pd.concat([MC_eval, SHERPA_df.loc[(SHERPA_df.loc[:, 'AUX_event'] % get_n_folds(DATASET_DIRPATH)).eq(fold_idx).to_numpy(), cat_cols]]).reset_index(drop=True)

        DATA_df, DATA_aux = get_test_subset_Dataframes(DATASET_DIRPATH, fold_idx, ['Data'], minimal=MINIMAL)
        for col in DATA_aux.columns: DATA_df.loc[:, col] = DATA_aux.loc[:, col]
        assert set(cat_cols) <= set(DATA_df.columns), f"The requested list of columns are not all in the Data file. Missing variables:\n{set(cat_cols) - set(DATA_df.columns)}"
        DATA_eval = DATA_df.loc[(DATA_df.loc[:, 'AUX_event'] % get_n_folds(DATASET_DIRPATH)).eq(fold_idx).to_numpy(), cat_cols].reset_index(drop=True)

        if OPT_SIDEBAND != 'none':
            sample_cut = np.zeros(MC_eval.shape[0])
            for sample_name in NONRES_MC_SAMPLENAMES:
                sample_cut = np.logical_or(sample_cut, MC_eval.loc[:, 'AUX_sample_name'].eq(sample_name).to_numpy())
            if OPT_SIDEBAND == 'mc':
                MC_eval.loc[:, 'cat_mask'] = np.where(~sample_cut, 'SR', 'SB')
                DATA_eval.loc[:, 'cat_mask'] = ''
                MC_eval.loc[MC_eval.loc[:, 'AUX_sample_name'].eq('GGJets').to_numpy(), 'AUX_eventWeight'] *= 1.59
                MC_eval.loc[
                    np.logical_or(
                        MC_eval.loc[:, 'AUX_sample_name'].eq('GJets').to_numpy(),
                        MC_eval.loc[:, 'AUX_sample_name'].eq('DDQCDGJets').to_numpy()
                    ), 'AUX_eventWeight'
                ] *= 1.21
            elif OPT_SIDEBAND == 'data':
                MC_eval.loc[:, 'cat_mask'] = np.where(~sample_cut, 'SR', '')
                DATA_eval.loc[:, 'cat_mask'] = 'SB'
            if len(table.field_names) == 0: 
                table.field_names = ['Category', 'FoM (s/b)'] + sorted(pd.unique(MC_eval['AUX_sample_name']).tolist()) + ['nonRes MC -- SB fit', 'Data -- SB fit']
        else: 
            MC_eval.loc[:, 'cat_mask'] = 'SR'
            DATA_eval.loc[:, 'cat_mask'] = ''
            if len(table.field_names) == 0: 
                table.field_names = ['Category', 'FoM (s/b)'] + sorted(pd.unique(MC_eval['AUX_sample_name']).tolist())

        df_eval = pd.concat([MC_eval, DATA_eval], join="inner").reset_index(drop=True)
        if full_df_eval is None: 
            full_df_eval = copy.deepcopy(df_eval)
        else: 
            full_df_eval = pd.concat([full_df_eval, df_eval], join="inner").reset_index(drop=True)

    full_category_mask = full_df_eval.loc[:, match_regex('AUX_*_resolved_BDT_mask', full_df_eval.columns)].eq(1).to_numpy()
    for cat_idx in range(1, CATEGORIZATION_OPTIONS['N_CATEGORIES']+1):
        categories_dict[f'cat{cat_idx}'] = {}
        
        best_fom, best_cut = CATEGORIZATION_METHOD(
            full_df_eval, full_category_mask, CATEGORIZATION_OPTIONS, TRANSFORM_CUT,
            [
                categories_dict[f'cat{prev_cat_idx}']['cut'] 
                for prev_cat_idx in range(1, cat_idx)
            ] if cat_idx != 1 else None, sideband_fit=OPT_SIDEBAND != 'none'
        )
        new_row = [f'Merged folds - Cat {cat_idx}', best_fom]
        for sample_name in table.field_names[2:]:
            if match_sample(sample_name, ['Data', 'MC']) is not None:
                pass_sideband_mask = np.logical_and(full_category_mask, categorization_utils.sideband_nonres_mask(full_df_eval, SB_str='SB' if match_sample(sample_name, [OPT_SIDEBAND]) is not None else ''))
                for i in range(len(TRANSFORM_COLUMNS)):
                    if '>' in TRANSFORM_CUT[i]: 
                        pass_sideband_mask = np.logical_and(pass_sideband_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                    else: 
                        pass_sideband_mask = np.logical_and(pass_sideband_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())

                def exp_func(x, a, b): return a * np.exp(b * x)
                _hist_ = hist.Hist(
                    hist.axis.Regular(int((categorization_utils.FIT_BINS[1]-categorization_utils.FIT_BINS[0])//categorization_utils.FIT_BINS[2]), categorization_utils.FIT_BINS[0], categorization_utils.FIT_BINS[1], name="var", growth=False, underflow=False, overflow=False), 
                    storage='weight'
                ).fill(var=full_df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), weight=full_df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy())
                params, _ = curve_fit(
                    exp_func, _hist_.axes.centers[0]-_hist_.axes.centers[0][0], _hist_.values(), p0=(_hist_.values()[0], -0.1), 
                    # sigma=np.where(_hist_.values() != 0, np.sqrt(_hist_.variances()), 0.76)
                )
                est_yield = quad(exp_func, categorization_utils.SR_CUTS[0]-_hist_.axes.centers[0][0], categorization_utils.SR_CUTS[1]-_hist_.axes.centers[0][0], args=tuple(params))[0] / categorization_utils.FIT_BINS[2]
                print('='*60+'\n'+'='*60)
                print(sample_name)
                print('-'*60)
                print(best_cut)
                categorization_utils.ascii_hist(
                    full_df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), 
                    bins=np.arange(categorization_utils.FIT_BINS[0], categorization_utils.FIT_BINS[1], categorization_utils.FIT_BINS[2]), 
                    weights=full_df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy(), 
                    fit=exp_func(_hist_.axes.centers[0]-_hist_.axes.centers[0][0], a=params[0], b=params[1])
                )
                print(f"y = {params[0]:.2f}e^({params[1]:.2f}x)")
                print(f"  -> est. yield of non-res in SR = {est_yield}, non-res bkg yield in SB = {np.sum(full_df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy())}")
                print('='*60+'\n'+'='*60)
                new_row.append(est_yield)
            else:
                if OPT_SIDEBAND == 'none' or sample_name not in NONRES_MC_SAMPLENAMES: SR_str = 'SR'
                elif OPT_SIDEBAND == 'mc' and sample_name in NONRES_MC_SAMPLENAMES: SR_str = 'SB'
                elif OPT_SIDEBAND == 'data' and sample_name in NONRES_MC_SAMPLENAMES: SR_str = ''
                pass_mask = np.logical_and(full_category_mask, categorization_utils.fom_mask(full_df_eval, SR_str=SR_str))
                for i in range(len(TRANSFORM_COLUMNS)):
                    if '>' in TRANSFORM_CUT[i]:
                        pass_mask = np.logical_and(pass_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                    else:
                        pass_mask = np.logical_and(pass_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())
                new_row.append(full_df_eval.loc[np.logical_and(pass_mask, full_df_eval.loc[:, 'AUX_sample_name'].eq(sample_name).to_numpy()), 'AUX_eventWeight'].sum())
        table.add_row(new_row)

        categories_dict[f'cat{cat_idx}']['fom'] = best_fom.item()
        categories_dict[f'cat{cat_idx}']['cut'] = best_cut.tolist()

        prev_category_mask = copy.deepcopy(full_category_mask)
        for i in range(len(TRANSFORM_COLUMNS)):
            if '>' in TRANSFORM_CUT[i]:
                prev_category_mask = np.logical_and(prev_category_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
            else:
                prev_category_mask = np.logical_and(prev_category_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())
        full_category_mask = np.logical_and(full_category_mask, ~prev_category_mask)
            
    if VERBOSE: print(' - '.join(os.path.normpath(TRAINING_DIRPATH).split('/')[-2:]+[DISCRIMINATOR])); table.float_format = '0.4'; print(table)

    if not MINIMAL:
        with open(CATEGORIZATION_FILEPATH, 'w') as f:
            json.dump(categories_dict, f)
        with open(CATEGORIZATION_FILEPATH.replace('.json', '_yields.csv'), 'w') as f:
            f.write(table.get_csv_string())


if __name__ == "__main__":
    categorize_model()