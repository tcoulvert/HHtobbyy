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


from retrieval_utils import (
    get_class_sample_map, get_n_folds,
    get_train_Dataframe
)
from training_utils import (
    get_dataset_dirpath,
)
from evaluation_utils import (
    get_filepaths, transform_preds_options, transform_preds_func
)

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath", 
    help="Full filepath on LPC for trained model files"
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
FOLD_TO_CATEGORIZE = args.fold
FOLD_TO_PLOT_OPTIONS = ['none', 'all']+[str(fold_idx) for fold_idx in range(get_n_folds(DATASET_DIRPATH))]
assert FOLD_TO_CATEGORIZE in FOLD_TO_PLOT_OPTIONS, f"The option passed to \'--fold\' ({FOLD_TO_CATEGORIZE}) is not allowed, for this dataset your options are: {FOLD_TO_PLOT_OPTIONS}"

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

TRANSFORM_LABELS, TRANSFORM_PREDS = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)
TRANSFORM_COLUMNS = [f"AUX_{transform_label}_prob" for transform_label in TRANSFORM_LABELS]
assert len(TRANSFORM_COLUMNS) <= 2, f"You're trying to run categorization over a discriminator with more than 2 dimensions, this likely won't converge with the brute-force way in this file. Please write your own categorization code or use a different discriminator"
CATEGORIZATION_COLUMNS = ['AUX_sample_name', 'AUX_eventWeight', 'AUX_mass', 'AUX_nonRes_resolved_BDT_mask', 'AUX_label1D'] + TRANSFORM_COLUMNS

CATEGORIZATION_DIRPATH = os.path.join(TRAINING_DIRPATH, "categorization", "")
if not os.path.exists(CATEGORIZATION_DIRPATH): 
    os.makedirs(CATEGORIZATION_DIRPATH)
CATEGORIZATION_FILEPATH = os.path.join(CATEGORIZATION_DIRPATH, f"{DISCRIMINATOR}{f'_sig{CLASS_NAMES[SIGNAL_LABEL]}' if SIGNAL_LABEL != 0 else ''}_categorization.json")
N_CATEGORIES = 3
N_STEPS = 50
N_ZOOM = 4

################################


def fom_mask(df: pd.DataFrame):
    return np.logical_and(df.loc[:, 'AUX_mass'].ge(122.5), df.loc[:, 'AUX_mass'].le(127.))

def sideband_nonres_mask(df: pd.DataFrame):
    return np.logical_and(
        np.logical_or(df.loc[:, 'AUX_mass'].lt(120.), df.loc[:, 'AUX_mass'].gt(130.)),
        np.logical_or(
            df.loc[:, 'AUX_sample_name'].eq('TTGG'),  # TTGG
            np.logical_or(
                np.logical_or(df.loc[:, 'AUX_sample_name'].eq('GJet'), df.loc[:, 'AUX_sample_name'].eq('GGJets')),  # GGJets or GJet
                df.loc[:, 'AUX_sample_name'].eq('DDQCDGJets')  # DDQCD GJet or GGJets
            )
        )
    )

def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)

def compute_cuts1D(df: pd.DataFrame, cat_mask: pd.DataFrame):
    pass_fom = np.logical_and(cat_mask, fom_mask(df))
    pass_sideband = np.logical_and(cat_mask, sideband_nonres_mask(df))

    best_fom, best_cut = 0., 0.

    start1, stop1 = 0., 1.
    for zoom in range(N_ZOOM):
        foms, cuts = [], []

        for cut1 in np.linspace(start1, stop1, N_STEPS, endpoint=True):
            cuts.append(cut1)

            pass_cut = np.logical_and(pass_fom, df.loc[:, TRANSFORM_COLUMNS[0]].gt(cut1))
            sideband_cut = np.logical_and(pass_sideband, df.loc[:, TRANSFORM_COLUMNS[0]].gt(cut1))

            signal = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].eq(SIGNAL_LABEL)), 'AUX_eventWeight'].sum()
            bkg = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].ne(SIGNAL_LABEL)), 'AUX_eventWeight'].sum()
            sideband_nonres = df.loc[sideband_cut, 'AUX_eventWeight'].sum()
            fom = fom_s_over_sqrt_b(signal, bkg) if sideband_nonres > 8. else 0.
            foms.append(fom)

        index = np.argmax(foms)
        fom, cut = foms[index], cuts[index]
        step_size1 = (stop1 - start1) / N_STEPS
        start1, stop1 = cut - step_size1, cut + step_size1

        if fom > best_fom: best_fom = fom; best_cut = cut

    return best_fom, best_cut

def compute_cuts2D(df: pd.DataFrame, cat_mask: pd.DataFrame):
    pass_fom = np.logical_and(cat_mask, fom_mask(df))
    pass_sideband = np.logical_and(cat_mask, sideband_nonres_mask(df))

    best_fom, best_cut = 0., (0., 0.)

    start1, stop1 = 0., 1.
    start2, stop2 = 0., 1.
    for zoom in range(N_ZOOM):
        foms, cuts = [], []
        
        for cut1 in np.linspace(start1, stop1, N_STEPS, endpoint=True):

            pass_cut = np.logical_and(pass_fom, df.loc[:, TRANSFORM_COLUMNS[0]].gt(cut1))
            sideband_cut = np.logical_and(pass_sideband, df.loc[:, TRANSFORM_COLUMNS[0]].gt(cut1))
            
            _foms_, _cuts_ = [], []
            for cut2 in np.linspace(start2, stop2, N_STEPS, endpoint=True):
                _cuts_.append( (cut1, cut2) )

                pass_cut = np.logical_and(pass_cut, df.loc[:, TRANSFORM_COLUMNS[1]].gt(cut2))
                sideband_cut = np.logical_and(pass_sideband, df.loc[:, TRANSFORM_COLUMNS[1]].gt(cut2))

                signal = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].eq(SIGNAL_LABEL)), 'AUX_eventWeight'].sum()
                bkg = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].ne(SIGNAL_LABEL)), 'AUX_eventWeight'].sum()
                sideband_nonres = df.loc[sideband_cut, 'AUX_eventWeight'].sum()
                fom = fom_s_over_sqrt_b(signal, bkg) if sideband_nonres > 8. and np.isfinite(fom_s_over_sqrt_b(signal, bkg)) else 0.
                # print(f"{_cuts_[-1]}: fom = {fom}; signal = {signal}, bkg = {bkg}, nonres sideband = {sideband_nonres}")
                _foms_.append(fom)
            foms.append(_foms_)
            cuts.append(_cuts_)
        
        index = np.unravel_index(np.argmax(foms), np.shape(foms))
        fom, cut = foms[index[0]][index[1]], cuts[index[0]][index[1]]
        step_size1, step_size2 = (stop1 - start1) / N_STEPS, (stop2 - start2) / N_STEPS
        start1, stop1 = cut[0] - step_size1, cut[0] + step_size1
        start2, stop2 = cut[1] - step_size2, cut[1] + step_size2

        if fom > best_fom: best_fom = fom; best_cut = cut

    print(f"{best_fom:.2f}, ({best_cut[0]:.4f}, {best_cut[1]:.4f})")
    return best_fom, best_cut

def categorize_model():
    categories_dict = {}

    full_MC_eval = None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        MC_df, MC_aux = get_train_Dataframe(DATASET_DIRPATH, fold_idx, dataset='test' if DATASET == 'train-test' else 'train')
        for col in MC_aux.columns: MC_df[col] = MC_aux[col]
        assert set(CATEGORIZATION_COLUMNS) <= set(MC_df.columns), f"The requested list of columns are not all in the file. Missing variables:\n{set(CATEGORIZATION_COLUMNS) - set(MC_df.columns)}"
        MC_eval = MC_df[CATEGORIZATION_COLUMNS]

        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
            if fold_idx == 0: full_MC_eval = copy.deepcopy(MC_eval)
            else: full_MC_eval = pd.concat([full_MC_eval, MC_eval]).reset_index(drop=True)
        if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == str(fold_idx):
            categories_dict[FOLD_TO_CATEGORIZE] = {}

            category_mask = np.logical_and(
                MC_eval.loc[:, 'AUX_nonRes_resolved_BDT_mask'].eq(1),
                MC_eval.loc[:, 'AUX_sample_name'].ne('Data')
            )
            for i in range(N_CATEGORIES):
                categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}'] = {}
                
                if len(TRANSFORM_COLUMNS) == 1:
                    best_fom, best_cut = compute_cuts1D(MC_eval, category_mask)
                    category_mask = np.logical_and(
                        category_mask, 
                        ~MC_eval.loc[:, TRANSFORM_COLUMNS[0]].gt(best_cut)
                    )
                else:
                    best_fom, best_cut = compute_cuts2D(MC_eval, category_mask)
                    category_mask = np.logical_and(
                        category_mask, 
                        ~np.logical_and(
                            MC_eval.loc[:, TRANSFORM_COLUMNS[0]].gt(best_cut[0]),
                            MC_eval.loc[:, TRANSFORM_COLUMNS[1]].gt(best_cut[1])
                        )
                    )

                categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}']['fom'] = best_fom
                categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}']['cut'] = best_cut

    if FOLD_TO_CATEGORIZE == 'all' or FOLD_TO_CATEGORIZE == 'none':
        categories_dict[FOLD_TO_CATEGORIZE] = {}

        category_mask = np.logical_and(
                full_MC_eval.loc[:, 'AUX_nonRes_resolved_BDT_mask'].eq(1),
                full_MC_eval.loc[:, 'AUX_sample_name'].ne('Data')
            )
        for i in range(N_CATEGORIES):
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}'] = {}
            
            if len(TRANSFORM_COLUMNS) == 1:
                best_fom, best_cut = compute_cuts1D(full_MC_eval, category_mask)
                category_mask = np.logical_and(
                    category_mask, 
                    ~full_MC_eval.loc[:, TRANSFORM_COLUMNS[0]].gt(best_cut)
                )
            else:
                best_fom, best_cut = compute_cuts2D(full_MC_eval, category_mask)
                category_mask = np.logical_and(
                    category_mask, 
                    ~np.logical_and(
                        full_MC_eval.loc[:, TRANSFORM_COLUMNS[0]].gt(best_cut[0]),
                        full_MC_eval.loc[:, TRANSFORM_COLUMNS[1]].gt(best_cut[1])
                    )
                )

            categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}']['fom'] = best_fom
            categories_dict[FOLD_TO_CATEGORIZE][f'cat{i}']['cut'] = best_cut

    with open(os.path.join(CATEGORIZATION_DIRPATH, f"categories.json"), 'w') as f:
        json.dump(categories_dict, f)


if __name__ == "__main__":
    categorize_model()