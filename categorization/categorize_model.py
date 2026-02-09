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
    "--fold",
    default='none',
    help="Make categories for specific fold. \'none\' only categorizes merged, \'all\' categorizes all folds separately and merged. Otherwise passing an int categorizes the fold with that index (index starting at 0)"
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
FOLD_TO_CATEGORIZE = args.fold
FOLD_TO_PLOT_OPTIONS = ['none', 'all']+[str(fold_idx) for fold_idx in range(get_n_folds(DATASET_DIRPATH))]
assert FOLD_TO_CATEGORIZE in FOLD_TO_PLOT_OPTIONS, f"The option passed to \'--fold\' ({FOLD_TO_CATEGORIZE}) is not allowed, for this dataset your options are: {FOLD_TO_PLOT_OPTIONS}"

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

TRANSFORM_LABELS, TRANSFORM_PREDS, TRANSFORM_CUT = transform_preds_func(CLASS_NAMES, DISCRIMINATOR, cutdirbool=True)
TRANSFORM_COLUMNS = [f"AUX_{transform_label}_prob" for transform_label in TRANSFORM_LABELS]
assert len(TRANSFORM_COLUMNS) <= 4, f"You're trying to run categorization over a discriminator with more than 4 dimensions, this likely won't converge with the brute-force way in this file. Please write your own categorization code or use a different discriminator"
CATEGORIZATION_COLUMNS = ['AUX_event', 'AUX_sample_name', 'AUX_eventWeight', 'AUX_mass', 'AUX_*_resolved_BDT_mask', 'AUX_label1D'] + TRANSFORM_COLUMNS

CATEGORIZATION_DIRPATH = os.path.join(TRAINING_DIRPATH, "categorization", "")
if not os.path.exists(CATEGORIZATION_DIRPATH): 
    os.makedirs(CATEGORIZATION_DIRPATH)
CATEGORIZATION_FILEPATH = os.path.join(CATEGORIZATION_DIRPATH, f"{DISCRIMINATOR}{f'_sig{CLASS_NAMES[SIGNAL_LABEL]}' if SIGNAL_LABEL != 0 else ''}_categorization.json")
CATEGORIZATION_OPTIONS_FILEPATH = os.path.join(CATEGORIZATION_DIRPATH, f"{DISCRIMINATOR}{f'_sig{CLASS_NAMES[SIGNAL_LABEL]}' if SIGNAL_LABEL != 0 else ''}_categorization_options.json")
with open(OPTIONS_FILEPATH, 'r') as f: CATEGORIZATION_OPTIONS = json.load(f)
CATEGORIZATION_OPTIONS['TRANSFORM_COLUMNS'] = TRANSFORM_COLUMNS
CATEGORIZATION_OPTIONS['SIGNAL_LABEL'] = SIGNAL_LABEL
CATEGORIZATION_METHOD = getattr(categorization_utils, CATEGORIZATION_OPTIONS['METHOD'])
if 'STARTSTOPS' not in CATEGORIZATION_OPTIONS.keys():
    CATEGORIZATION_OPTIONS['STARTSTOPS'] = [[0., 1.] if '<' in TRANSFORM_CUT[i] else [1., 0.] for i in range(len(TRANSFORM_COLUMNS))]

################################


def categorize_model():
    categories_dict = {}
    table = pt.PrettyTable()

    full_df_eval = None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        if FOLD_TO_CATEGORIZE not in ['all', 'none'] and FOLD_TO_CATEGORIZE != str(fold_idx): continue
        MC_df, MC_aux = get_train_Dataframe(DATASET_DIRPATH, fold_idx, dataset='test' if DATASET == 'train-test' else 'train', minimal=MINIMAL)
        for col in MC_aux.columns: MC_df.loc[:, col] = MC_aux.loc[:, col]
        cat_cols = [match_regex(cat_col, MC_df.columns) for cat_col in CATEGORIZATION_COLUMNS]
        assert set(cat_cols) <= set(MC_df.columns), f"The requested list of columns are not all in the file. Missing variables:\n{set(cat_cols) - set(MC_df.columns)}"
        MC_eval = MC_df.loc[:, cat_cols]

        DATA_df, DATA_aux = get_test_subset_Dataframes(DATASET_DIRPATH, fold_idx, ['!2024*Data'], minimal=MINIMAL)
        DATA_aux['AUX_eventWeight'] = DATA_aux['AUX_eventWeight'] * 2.76  # lumi 22-24 / lumi 22-23
        # DATA_df, DATA_aux = get_test_subset_Dataframes(DATASET_DIRPATH, fold_idx, ['Data'], minimal=MINIMAL)
        for col in DATA_aux.columns: DATA_df.loc[:, col] = DATA_aux.loc[:, col]
        assert set(cat_cols) <= set(DATA_df.columns), f"The requested list of columns are not all in the Data file. Missing variables:\n{set(cat_cols) - set(DATA_df.columns)}"
        DATA_eval = DATA_df.loc[(DATA_df.loc[:, 'AUX_event'] % get_n_folds(DATASET_DIRPATH)).eq(fold_idx).to_numpy(), cat_cols]

        if OPT_SIDEBAND != 'none':
            sample_cut = np.logical_or(
                MC_eval.loc[:, 'AUX_sample_name'].eq('TTGG').to_numpy(),  # TTGG
                np.logical_or(
                    np.logical_or(MC_eval.loc[:, 'AUX_sample_name'].eq('GJet').to_numpy(), MC_eval.loc[:, 'AUX_sample_name'].eq('GGJets').to_numpy()),  # GGJets or GJet
                    MC_eval.loc[:, 'AUX_sample_name'].eq('DDQCDGJets').to_numpy()  # DDQCD GJet or GGJets
                )
            )
            if OPT_SIDEBAND == 'mc':
                MC_eval.loc[:, 'cat_mask'] = np.where(~sample_cut, 'SR', 'SB')
                DATA_eval.loc[:, 'cat_mask'] = ''
            elif OPT_SIDEBAND == 'data':
                MC_eval.loc[:, 'cat_mask'] = np.where(~sample_cut, 'SR', '')
                DATA_eval.loc[:, 'cat_mask'] = 'SB'
        else: 
            MC_eval.loc[:, 'cat_mask'] = 'SR'
            DATA_eval.loc[:, 'cat_mask'] = ''

        df_eval = pd.concat([MC_eval, DATA_eval], join="inner").reset_index(drop=True)
        
        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
            if full_df_eval is None: 
                full_df_eval = copy.deepcopy(df_eval)
            else: 
                full_df_eval = pd.concat([full_df_eval, df_eval], join="inner").reset_index(drop=True)
                # full_df_eval = pd.concat([full_df_eval, MC_eval], join="inner").reset_index(drop=True)
                # data_mask = full_df_eval.loc[:, 'AUX_sample_name'].eq('Data').to_numpy()
                # # print(np.all(full_df_eval.loc[data_mask, 'AUX_mass'].to_numpy() == DATA_eval.loc[:, 'AUX_mass'].to_numpy()))
                # for col in TRANSFORM_COLUMNS: 
                #     full_df_eval.loc[data_mask, col] += DATA_eval.loc[:, col]

        if len(table.field_names) == 0: table.field_names = ['Category', 'FoM (s/b)'] + sorted(pd.unique(MC_eval['AUX_sample_name']).tolist())
        
        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == str(fold_idx):
            categories_dict[FOLD_TO_CATEGORIZE] = {}

            category_mask = df_eval.loc[:, match_regex('AUX_*_resolved_BDT_mask', df_eval.columns)].eq(1).to_numpy()

            for cat_idx in range(1, CATEGORIZATION_OPTIONS['N_CATEGORIES']+1):
                categories_dict[str(fold_idx)][f'cat{cat_idx}'] = {}
                
                best_fom, best_cut = CATEGORIZATION_METHOD(
                    df_eval, category_mask, CATEGORIZATION_OPTIONS, TRANSFORM_CUT, 
                    [
                        categories_dict[FOLD_TO_CATEGORIZE][f'cat{prev_cat_idx}']['cut'] 
                        for prev_cat_idx in range(1, cat_idx)
                    ] if cat_idx != 1 else None
                )
                new_row = [f'Merged folds - Cat {cat_idx}', best_fom]
                pass_mask = np.logical_and(category_mask, categorization_utils.fom_mask(df_eval))
                for i in range(len(TRANSFORM_COLUMNS)):
                    if '>' in TRANSFORM_CUT[i]:
                        pass_mask = np.logical_and(pass_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                    else:
                        pass_mask = np.logical_and(pass_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())
                for sample_name in table.field_names[2:]:
                    if match_sample(sample_name, ['Data']) is not None:
                        sideband_mask = np.logical_and(category_mask, categorization_utils.sideband_nonres_mask(df_eval))
                        for i in range(len(TRANSFORM_COLUMNS)):
                            if '>' in TRANSFORM_CUT[i]:
                                pass_sideband_mask = np.logical_and(sideband_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                            else:
                                pass_sideband_mask = np.logical_and(sideband_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())

                        def exp_func(x, a, b): return a * np.exp(b * x)
                        _hist_, _bins_ = np.histogram(
                            df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), 
                            bins=np.arange(100., 180., 5.), 
                            weights=df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy()
                        )
                        params, _ = curve_fit(
                            exp_func, _bins_[:-1]-_bins_[0], _hist_, p0=(_hist_[0], -0.1), 
                            sigma=np.where(np.isfinite(_hist_**-1), _hist_**-1, 0.76)
                        )
                        est_yield, _ = quad(exp_func, categorization_utils.SR_CUTS[0], categorization_utils.SR_CUTS[1], args=tuple(params))
                        print('='*60+'\n'+'='*60)
                        print(best_cut)
                        categorization_utils.ascii_hist(
                            df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), 
                            bins=np.arange(100., 180., 5.), 
                            weights=df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy(), 
                            fit=exp_func(_bins_[:-1]-_bins_[0], a=params[0], b=params[1])
                        )
                        print(f"y = {params[0]:.2f}e^({params[1]:.2f}x)")
                        print(f"  -> est. yield of non-res in SR = {est_yield}")
                        print('='*60+'\n'+'='*60)
                        new_row.append(est_yield)
                    else:
                        new_row.append(df_eval.loc[np.logical_and(pass_mask, df_eval.loc[:, 'AUX_sample_name'].eq(sample_name).to_numpy()), 'AUX_eventWeight'].sum())
                table.add_row(new_row)                
                prev_category_mask = copy.deepcopy(category_mask)
                for i in range(len(TRANSFORM_COLUMNS)):
                    if '>' in TRANSFORM_CUT[i]:
                        prev_category_mask = np.logical_and(prev_category_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                    else:
                        prev_category_mask = np.logical_and(prev_category_mask, df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())
                category_mask = np.logical_and(category_mask, ~prev_category_mask)

                categories_dict[str(fold_idx)][f'cat{cat_idx}']['fom'] = best_fom.item()
                categories_dict[str(fold_idx)][f'cat{cat_idx}']['cut'] = best_cut.tolist()

    # for col in TRANSFORM_COLUMNS: 
    #     full_df_eval.loc[full_df_eval.loc[:, 'AUX_sample_name'].eq('Data'), col] /= get_n_folds(DATASET_DIRPATH)

    if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
        categories_dict[FOLD_TO_CATEGORIZE] = {}

        full_category_mask = full_df_eval.loc[:, match_regex('AUX_*_resolved_BDT_mask', full_df_eval.columns)].eq(1).to_numpy()

        for cat_idx in range(1, CATEGORIZATION_OPTIONS['N_CATEGORIES']+1):
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}'] = {}
            
            best_fom, best_cut = CATEGORIZATION_METHOD(
                full_df_eval, full_category_mask, CATEGORIZATION_OPTIONS, TRANSFORM_CUT,
                [
                    categories_dict[FOLD_TO_CATEGORIZE][f'cat{prev_cat_idx}']['cut'] 
                    for prev_cat_idx in range(1, cat_idx)
                ] if cat_idx != 1 else None
            )
            new_row = [f'Merged folds - Cat {cat_idx}', best_fom]
            pass_mask = np.logical_and(full_category_mask, categorization_utils.fom_mask(full_df_eval))
            for i in range(len(TRANSFORM_COLUMNS)):
                if '>' in TRANSFORM_CUT[i]:
                    pass_mask = np.logical_and(pass_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                else:
                    pass_mask = np.logical_and(pass_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())
            for sample_name in table.field_names[2:]:
                if match_sample(sample_name, ['Data']) is not None:
                    pass_sideband_mask = np.logical_and(full_category_mask, categorization_utils.sideband_nonres_mask(full_df_eval))
                    for i in range(len(TRANSFORM_COLUMNS)):
                        if '>' in TRANSFORM_CUT[i]: 
                            pass_sideband_mask = np.logical_and(pass_sideband_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy())
                        else: 
                            pass_sideband_mask = np.logical_and(pass_sideband_mask, full_df_eval.loc[:, TRANSFORM_COLUMNS[i]].lt(best_cut[i]).to_numpy())

                    def exp_func(x, a, b): return a * np.exp(b * x)
                    _hist_, _bins_ = np.histogram(
                        full_df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), 
                        bins=np.arange(100., 180., 5.), 
                        weights=full_df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy()
                    )
                    params, _ = curve_fit(
                        exp_func, _bins_[:-1]-_bins_[0], _hist_, p0=(_hist_[0], -0.1), 
                        sigma=np.where(np.isfinite(_hist_**-1), _hist_**-1, 0.76)
                    )
                    est_yield, _ = quad(exp_func, categorization_utils.SR_CUTS[0], categorization_utils.SR_CUTS[1], args=tuple(params))
                    print('='*60+'\n'+'='*60)
                    print(best_cut)
                    categorization_utils.ascii_hist(
                        full_df_eval.loc[pass_sideband_mask, 'AUX_mass'].to_numpy(), 
                        bins=np.arange(100., 180., 5.), 
                        weights=full_df_eval.loc[pass_sideband_mask, 'AUX_eventWeight'].to_numpy(), 
                        fit=exp_func(_bins_[:-1]-_bins_[0], a=params[0], b=params[1])
                    )
                    print(f"y = {params[0]:.2f}e^({params[1]:.2f}x)")
                    print(f"  -> est. yield of non-res in SR = {est_yield}")
                    print('='*60+'\n'+'='*60)
                    new_row.append(est_yield)
                else:
                    new_row.append(full_df_eval.loc[np.logical_and(pass_mask, full_df_eval.loc[:, 'AUX_sample_name'].eq(sample_name).to_numpy()), 'AUX_eventWeight'].sum())
            table.add_row(new_row)

            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['fom'] = best_fom.item()
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['cut'] = best_cut.tolist()

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