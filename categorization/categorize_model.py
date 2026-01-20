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

from preprocessing_utils import match_regex
from retrieval_utils import (
    get_class_sample_map, get_n_folds,
    get_train_Dataframe
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
OPTIONS_FILEPATH = args.options_filepath
FOLD_TO_CATEGORIZE = args.fold
FOLD_TO_PLOT_OPTIONS = ['none', 'all']+[str(fold_idx) for fold_idx in range(get_n_folds(DATASET_DIRPATH))]
assert FOLD_TO_CATEGORIZE in FOLD_TO_PLOT_OPTIONS, f"The option passed to \'--fold\' ({FOLD_TO_CATEGORIZE}) is not allowed, for this dataset your options are: {FOLD_TO_PLOT_OPTIONS}"

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

TRANSFORM_LABELS, TRANSFORM_PREDS = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)
TRANSFORM_COLUMNS = [f"AUX_{transform_label}_prob" for transform_label in TRANSFORM_LABELS]
assert len(TRANSFORM_COLUMNS) <= 4, f"You're trying to run categorization over a discriminator with more than 4 dimensions, this likely won't converge with the brute-force way in this file. Please write your own categorization code or use a different discriminator"
CATEGORIZATION_COLUMNS = ['AUX_sample_name', 'AUX_eventWeight', 'AUX_mass', 'AUX_*_resolved_BDT_mask', 'AUX_label1D'] + TRANSFORM_COLUMNS

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
    CATEGORIZATION_OPTIONS['STARTSTOPS'] = [[0., 1.] for _ in TRANSFORM_COLUMNS]

################################


def categorize_model():
    categories_dict = {}

    full_MC_eval = None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        MC_df, MC_aux = get_train_Dataframe(DATASET_DIRPATH, fold_idx, dataset='test' if DATASET == 'train-test' else 'train')
        for col in MC_aux.columns: MC_df[col] = MC_aux[col]
        cat_cols = [match_regex(cat_col, MC_df.columns) for cat_col in CATEGORIZATION_COLUMNS]
        assert set(cat_cols) <= set(MC_df.columns), f"The requested list of columns are not all in the file. Missing variables:\n{set(cat_cols) - set(MC_df.columns)}"
        MC_eval = MC_df[cat_cols]

        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
            if fold_idx == 0: full_MC_eval = copy.deepcopy(MC_eval)
            else: full_MC_eval = pd.concat([full_MC_eval, MC_eval]).reset_index(drop=True)
        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == str(fold_idx):
            categories_dict[FOLD_TO_CATEGORIZE] = {}

            category_mask = MC_eval.loc[:, match_regex('AUX_*_resolved_BDT_mask', MC_eval.columns)].eq(1).to_numpy()
            for cat_idx in range(1, CATEGORIZATION_OPTIONS['N_CATEGORIES']+1):
                categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}'] = {}
                
                best_fom, best_cut = CATEGORIZATION_METHOD(MC_eval, category_mask, CATEGORIZATION_OPTIONS)
                prev_category_mask = copy.deepcopy(category_mask)
                for i in range(len(TRANSFORM_COLUMNS)):
                    prev_category_mask = np.logical_and(
                        prev_category_mask, MC_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i])
                    )
                category_mask = np.logical_and(category_mask, ~prev_category_mask)

                categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['fom'] = best_fom
                categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['cut'] = best_cut

    if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
        categories_dict[FOLD_TO_CATEGORIZE] = {}

        category_mask = full_MC_eval.loc[:, match_regex('AUX_*_resolved_BDT_mask', full_MC_eval.columns)].eq(1).to_numpy()

        for cat_idx in range(1, CATEGORIZATION_OPTIONS['N_CATEGORIES']+1):
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}'] = {}
            
            best_fom, best_cut = CATEGORIZATION_METHOD(full_MC_eval, category_mask, CATEGORIZATION_OPTIONS)
            print(f"Best fom = {best_fom}, best cut = {best_cut}")

            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['fom'] = best_fom
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx}']['cut'] = best_cut
            
            if VERBOSE:
                print('='*60+'\n'+'='*60)
                print('/'.join(os.path.normpath(TRAINING_DIRPATH).split('/')[-2:]))
                print(f"cat{cat_idx} yields")
                print('-'*60)
                print_str = ' & '.join([f"{TRANSFORM_COLUMNS[i]} > {best_cut[i]:.4f}" for i in range(len(TRANSFORM_COLUMNS))])
                if cat_idx > 1:
                    print_str = print_str + ' & '.join(['']+[f"{TRANSFORM_COLUMNS[i]} ≤ {categories_dict[FOLD_TO_CATEGORIZE][f'cat{cat_idx-1}']['cut'][i]:.4f}" for i in range(len(TRANSFORM_COLUMNS))])
                print(print_str)
                print('-'*60)
                pass_mask = np.logical_and(
                    category_mask, categorization_utils.fom_mask(full_MC_eval)
                )
                for i in range(len(TRANSFORM_COLUMNS)):
                    pass_mask = np.logical_and(
                        pass_mask, MC_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy()
                    )
                for unique_label in np.unique(full_MC_eval.loc[:, 'AUX_sample_name']):
                    unique_yield = full_MC_eval.loc[np.logical_and(pass_mask, full_MC_eval.loc[:, 'AUX_sample_name'].eq(unique_label).to_numpy()), 'AUX_eventWeight'].sum()
                    print('-'*60)
                    print(f"{unique_label} yield = {unique_yield:.4f}")

            prev_category_mask = copy.deepcopy(category_mask)
            for i in range(len(TRANSFORM_COLUMNS)):
                prev_category_mask = np.logical_and(
                    prev_category_mask, full_MC_eval.loc[:, TRANSFORM_COLUMNS[i]].gt(best_cut[i]).to_numpy()
                )
            category_mask = np.logical_and(category_mask, ~prev_category_mask)
            

    with open(CATEGORIZATION_FILEPATH, 'w') as f:
        json.dump(categories_dict, f)


if __name__ == "__main__":
    categorize_model()