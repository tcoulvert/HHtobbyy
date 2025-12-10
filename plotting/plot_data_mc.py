# Stdlib packages
import argparse
import copy
import logging
import os
import subprocess
import sys

# Common Py packages
import matplotlib.pyplot as plt

# HEP packages
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

# Module packages
from plot_hists import (
    plot_1dhist, plot_ratio
)
from plotting_utils import combine_prepostfix
from retrieval_utils import (
    get_class_sample_map, get_n_folds,
    get_Dataframe
)
from training_utils import (
    get_dataset_dirpath, get_model_func
)
from evaluation_utils import (
    transform_preds_options, transform_preds_func,
    get_filepaths
)

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath",
    help="Full filepath for trained model files"
)
parser.add_argument(
    "--dataset_dirpath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_dirpath.txt` file"
)
parser.add_argument(
    "--dataset", 
    choices=["train", "train-test", "test"], 
    default="test",
    help="Make output score distributions for what dataset"
)
parser.add_argument(
    "--density", 
    action="store_true",
    help="Boolean to make plots density"
)
parser.add_argument(
    "--liny", 
    action="store_true",
    help="Boolean to make plots linear-scale on y-axis"
)
parser.add_argument(
    "--syst_name", 
    default="nominal",
    help="Name of systemaic to evaluate sculpting over, generally this should just stay as \"nominal\""
)
parser.add_argument(
    "--seed", 
    default=21,
    help="Seed for RNG"
)
parser.add_argument(
    "--discriminator", 
    choices=transform_preds_options(),
    default=None,
    help="Defines the discriminator to use for output scores, discriminators are implemented in evaluation_utils"
)

################################


args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
if args.dataset_dirpath is None:
    DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
else:
    DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')
DATASET = args.dataset
DENSITY = args.density
LINY = args.liny
SYST_NAME = args.syst_name
SEED = args.seed
DISCRIMINATOR = args.discriminator

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]
PLOT_TYPE = f"Data_MC_{DATASET}"

################################


def sideband_cuts(aux_: pd.DataFrame):
    event_mask = np.logical_and(
        np.logical_or(aux['AUX_mass'] < 110, aux['AUX_mass'] > 140),
        aux['AUX_nonRes_resolved_BDT_mask']
    )
    return event_mask

def data_mc_comparison():
    get_booster = get_model_func(TRAINING_DIRPATH)

    if DISCRIMINATOR is not None:
        transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)

    get_filepaths_func = get_filepaths(DATASET_DIRPATH, DATASET, SYST_NAME)

    full_Data_df, full_Data_aux = None, None
    full_MC_dfs, full_MC_auxs, full_MC_labels = None, None, None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        booster = get_booster(fold_idx)

        Data_filepaths = [filepath for filepath in get_filepaths_func(fold_idx)[DATASET] if 'data' in filepath.lower()]
        MC_filepaths = [filepath for filepath in get_filepaths_func(fold_idx)[DATASET] if 'sim' in filepath.lower()]
        assert len(set(get_filepaths_func(fold_idx)[DATASET])) == len(set(Data_filepaths) | set(MC_filepaths)), f"Some files are not being found as data or sim, please check that data filepaths contain \"data\" (not case sensitive) and sim filepaths contain \"sim\""
        
        Data_df, Data_aux = pd.concat([get_Dataframe(filepath) for filepath in Data_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in Data_filepaths])
        MC_dfs, MC_auxs = [get_Dataframe(filepath) for filepath in MC_filepaths], [get_Dataframe(filepath, aux=True) for filepath in MC_filepaths]
        MC_labels = [MC_aux.loc[:, 'AUX_sample_name'][0] for MC_aux in MC_auxs]
        unique_MC_labels = np.unique(MC_labels)
        unique_MC_dfs, unique_MC_auxs = [], []
        for unique_MC_label in unique_MC_labels:
            unique_MC_dfs.append(pd.concat([MC_dfs[i] for i in range(len(MC_labels)) if MC_labels[i] == unique_MC_label]))
            unique_MC_auxs.append(pd.concat([MC_auxs[i] for i in range(len(MC_labels)) if MC_labels[i] == unique_MC_label]))
        MC_dfs, MC_auxs, MC_labels = unique_MC_dfs, unique_MC_auxs, unique_MC_labels

        Data_mask, MC_masks = sideband_cuts(Data_aux), [sideband_cuts(MC_aux) for MC_aux in MC_auxs]

        if fold_idx == 0:
            full_Data_df, full_Data_aux = copy.deepcopy(Data_df.loc[Data_mask]), copy.deepcopy(Data_aux.loc[Data_mask])
            full_MC_dfs, full_MC_auxs, full_MC_labels = (
                copy.deepcopy([MC_dfs[i].loc[MC_masks[i]] for i in range(len(MC_labels))]),
                copy.deepcopy([MC_auxs[i].loc[MC_masks[i]] for i in range(len(MC_labels))]),
                MC_labels
            )
        else:
            full_Data_df, full_Data_aux = (
                pd.concat([full_Data_df, Data_df.loc[Data_mask]]).reset_index(drop=True), 
                pd.concat([full_Data_aux, Data_aux.loc[Data_mask]]).reset_index(drop=True)
            )
            full_MC_dfs, full_MC_auxs = (
                [pd.concat([full_MC_dfs[i], MC_dfs[i].loc[MC_masks[i]]]) for i in range(len(MC_labels))],
                [pd.concat([full_MC_auxs[i], MC_auxs[i].loc[MC_masks[i]]]) for i in range(len(MC_labels))]
            )

        assert all(all(sorted(Data_df.columns[i]) == sorted(MC_df.columns[i]) for i in range(len(Data_df.columns))) for MC_df in MC_dfs), f"Data and MC have different variables"
        for col in Data_df.columns:
            fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8))

            Data_arr = Data_df.loc[Data_mask, col].to_numpy()
            plot_1dhist(
                Data_arr, TRAINING_DIRPATH, PLOT_TYPE, col,
                weights=np.ones_like(Data_arr), yerr=True, subplots=(fig, axs[0]),
                histtype="errorbar", labels='Data', logy=not LINY, density=DENSITY,
                colors='k'
            )

            MC_arrs = [MC_dfs[i].loc[MC_masks[i], col].to_numpy() for i in range(len(MC_dfs))]
            MC_weights = [MC_auxs[i].loc[MC_masks[i], 'AUX_eventWeight'].to_numpy() for i in range(len(MC_dfs))]
            plot_1dhist(
                MC_arrs, TRAINING_DIRPATH, PLOT_TYPE, col,
                weights=MC_weights, yerr=True, subplots=(fig, axs[0]),
                histtype='step' if DENSITY else 'fill', labels=MC_labels, 
                logy=not LINY, density=DENSITY, stack=not DENSITY,
            )

            plot_ratio(
                Data_arr, MC_arrs, col, subplots=(fig, axs), 
                training_dirpath=TRAINING_DIRPATH, plot_type=PLOT_TYPE,
                weights1=np.ones_like(Data_arr), weights2=MC_weights,
                save_and_close=True,
                plot_prefix=col, 
                plot_postfix=combine_prepostfix(f"fold{fold_idx}", 'density' if DENSITY else '', 'postfix')
            )

    assert all(all(sorted(full_Data_df.columns[i]) == sorted(MC_df.columns[i]) for i in range(len(Data_df.columns))) for MC_df in full_MC_dfs), f"Data and MC have different variables"
    for col in full_Data_df.columns:
        fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8))

        Data_arr = full_Data_df.loc[:, col].to_numpy()
        plot_1dhist(
            Data_arr, TRAINING_DIRPATH, PLOT_TYPE, col,
            weights=np.ones_like(Data_arr), yerr=True, subplots=(fig, axs[0]),
            histtype="errorbar", labels='Data', logy=not LINY, density=DENSITY,
            colors='k'
        )

        MC_arrs = [full_MC_dfs[i].loc[:, col].to_numpy() for i in range(len(full_MC_labels))]
        MC_weights = [full_MC_auxs[i].loc[:, 'AUX_eventWeight'].to_numpy() for i in range(len(full_MC_labels))]
        plot_1dhist(
            MC_arrs, TRAINING_DIRPATH, PLOT_TYPE, col,
            weights=MC_weights, yerr=True, subplots=(fig, axs[0]),
            histtype='step' if DENSITY else 'fill', labels=MC_labels, 
            logy=not LINY, density=DENSITY, stack=not DENSITY,
        )

        plot_ratio(
            Data_arr, MC_arrs, col, subplots=(fig, axs), 
            training_dirpath=TRAINING_DIRPATH, plot_type=PLOT_TYPE,
            weights1=np.ones_like(Data_arr), weights2=MC_weights,
            yerr=True, save_and_close=True,
            plot_prefix=col, plot_postfix='density' if DENSITY else ''
        )
            
        
if __name__ == "__main__":
    data_mc_comparison()