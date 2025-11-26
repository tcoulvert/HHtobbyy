# Stdlib packages
import argparse
import copy
import json
import logging
import os
import subprocess
import sys

# Common Py packages
import matplotlib.pyplot as plt

# HEP packages
import numpy as np
import pandas as pd
import scipy as scp

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
    plot_1dhist
)
from retrieval_utils import (
    get_class_sample_map, get_n_folds, 
    match_sample, match_regex,
    get_Dataframe, get_DMatrix
)
from training_utils import (
    get_dataset_dirpath, get_model_func
)
from evaluation_utils import (
    evaluate, transform_preds_options, transform_preds_func,
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
    "sculpting_cuts",
    help="JSON file that defines the discriminator(s) to use, the cuts to apply, and other configurations for the sculpting check"
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
    help="Make output score distributions for what dataset"
)
parser.add_argument(
    "--weights", 
    action="store_true",
    help="Boolean to make plots using MC weights"
)
parser.add_argument(
    "--density", 
    action="store_true",
    help="Boolean to make plots density"
)
parser.add_argument(
    "--logy", 
    action="store_true",
    help="Boolean to make plots log-scale on y-axis"
)
parser.add_argument(
    "--resample",
    type=int,
    default=10,
    help="Number of times to resample each event to try and get a \"good\" event"
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
    "--fit", 
    action="store_true",
    help="Boolean to make exponential + gaussian fit"
)

################################


args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
if args.dataset_dirpath is None:
    DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
else:
    DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')
DATASET = args.dataset
WEIGHTS = args.weights
DENSITY = args.density
LOGY = args.logy
RESAMPLE = args.resample
SYST_NAME = args.syst_name
SEED = args.seed
FIT = args.fit

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]
PLOT_TYPE = "resample_sculpting"

with open(args.sculpting_cuts, 'r') as f:
    SCULPTING_CUTS = json.load(f)
DISCRIMINATOR = SCULPTING_CUTS.pop('discriminator')
assert DISCRIMINATOR in transform_preds_options(), f"Trying to use a discriminator ({DISCRIMINATOR}) that isn't implemented in evaluation_utils. Use one of {transform_preds_options()} or implement your own"
NONRES_SAMPLES = SCULPTING_CUTS.pop('nonRes_samples')
PLOT_VARS = SCULPTING_CUTS.pop('plot_vars')

################################


def resample_from_var(
    arr, weight, n_events, min_value=None, bins=100
):
    resample_rng = np.random.default_rng(seed=SEED)

    np_hist, bin_edges = np.histogram(arr, bins=bins, weights=weight, density=True)
    np_hist /= np.sum(np_hist)

    bin_choices = resample_rng.choice(np.arange(len(np_hist)), size=n_events, p=np_hist)

    value_choices = (bin_edges[bin_choices+1] - bin_edges[bin_choices]) * resample_rng.random(size=n_events) + bin_edges[bin_choices]

    if min_value is None or np.all(value_choices > min_value):
        return value_choices
    else:  # this is not really correct, just an approximation to make the code work faster
        bad_choices_bool = value_choices <= min_value

        largest_min_value = np.max(min_value[bad_choices_bool])

        np_hist, bin_edges = np.histogram(arr[arr > largest_min_value], bins=bins, weights=arr[arr > largest_min_value], density=True)
        np_hist /= np.sum(np_hist)

        bin_choices = resample_rng.choice(np.arange(len(np_hist)), size=np.sum(bad_choices_bool), p=np_hist)
        
        value_choices[bad_choices_bool] = (bin_edges[bin_choices+1] - bin_edges[bin_choices]) * resample_rng.random(size=np.sum(bad_choices_bool)) + bin_edges[bin_choices]
        
        return value_choices

def resample_grow_pd(var, n_duplicates_per_event):
    new_rows = pd.DataFrame(
        np.tile(var.to_numpy(), (n_duplicates_per_event, 1)), columns=var.columns
    )
    new_rows['AUX_resampled'] = np.ones_like(new_rows['AUX_hash'].to_numpy(), dtype=bool)
    return pd.concat([var, new_rows], ignore_index=True)


def exp_plus_gauss(x, A, tau, B, sigma, C):
    exp = A * np.exp(-x * tau)
    gauss = B * np.exp(-0.5 * ((x - 125) / sigma)**2)
    return exp + gauss + C

def fit_tightest(np_arr, plot_var, subplots):

    np_hist, bin_edges = np.histogram(np_arr, bins=plot_var['bins'], range=plot_var['range'], density=True)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(np_hist))]

    popt, pcov = scp.optimize.curve_fit(exp_plus_gauss, bin_centers, np_hist, p0=[10, 1/70, 1, 2, 0])
    perr = np.sqrt(np.diag(pcov))

    for i, param in enumerate(['Exp Amp', 'Exp tau', 'Gauss Amp', 'Gauss sigma', 'Y-intercept']):
        print(f"{param} = {popt[i]:.4f}±{perr[i]:.4f}")

    x_trial = np.linspace(100, 180, 1000)
    y_fit = exp_plus_gauss(x_trial, *popt)

    fig, ax = subplots
    ax.plot(x_trial, y_fit, color='blue', label=f"Fit - Exponential + Gaussian@125GeV (A={popt[2]:.2f}±{perr[2]:.2f})")
    return fig, ax


def sculpting_check():
    get_booster = get_model_func(TRAINING_DIRPATH)

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)

    hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}

    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        booster = get_booster(fold_idx)

        get_resample_filepaths = lambda nonres_samples_key: [
            filepath for filepath in get_filepaths(DATASET_DIRPATH, DATASET, SYST_NAME)(fold_idx) 
            if match_sample(filepath, [sample[nonres_samples_key] for sample in NONRES_SAMPLES]) is not None
        ]
        nonRes_filepaths = get_resample_filepaths('name')
        nonRes_df, nonRes_aux = pd.concat([get_Dataframe(filepath) for filepath in nonRes_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in nonRes_filepaths])
        nonRes_aux['AUX_resampled'] = np.zeros_like(nonRes_aux['AUX_hash'].to_numpy(), dtype=bool)
        nonRes_df, nonRes_aux = resample_grow_pd(nonRes_df, RESAMPLE), resample_grow_pd(nonRes_aux, RESAMPLE)

        Res_filepaths = get_resample_filepaths('resample_from')
        Res_df, Res_aux = pd.concat([get_Dataframe(filepath) for filepath in Res_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in Res_filepaths])

        fold_hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}
        while True:

            for nonRes_sample in NONRES_SAMPLES:
                nonRes_mask = np.logical_and(
                    nonRes_aux.loc[:, "AUX_sample_name"] == nonRes_sample["name"],
                    nonRes_aux.loc[:, "AUX_resampled"]
                )
                Res_mask = (Res_aux.loc[:, "AUX_sample_name"] == nonRes_sample["resample_from"])

                for resample_var in nonRes_sample["resample_vars"]:
                    variable = match_regex(resample_var, nonRes_df.columns)
                    assert variable is not None, f"Variable with regex string {resample_var} does not exist in Dataframe, check if the regex string is correct"
                    nonRes_df.loc[nonRes_mask, variable] = resample_from_var(
                        Res_df.loc[Res_mask, variable], weight=Res_aux.loc[Res_mask, "AUX_eventWeight"] if WEIGHTS else np.ones(np.sum(Res_mask)),
                        n_events=np.sum(nonRes_mask), min_value=nonRes_df.loc[nonRes_mask, "AUX_max_nonbjet_btag"] if "btag" in variable and "WP" not in variable else None
                    )

            nonRes_dm = get_DMatrix(nonRes_df, nonRes_aux, dataset='test', label=False)
            nonRes_preds = evaluate(booster, nonRes_dm)
            transformed_preds = transform_preds(nonRes_preds)

            for hist_name, cut_dict in SCULPTING_CUTS.items():

                cut_mask = np.ones(np.shape(transformed_preds)[0], dtype=bool)
                for cut_axis, cut_range in cut_dict.items():
                    cut_mask = np.logical_and(
                        cut_mask, 
                        np.logical_and(
                            transform_preds[:, int(cut_axis)] > cut_range[0],
                            transform_preds[:, int(cut_axis)] < cut_range[1],
                        )
                    )
                if np.sum(cut_mask) == 0: break

                pass_cut_df = pd.concat([
                    nonRes_df.loc[cut_mask, [match_regex(plot_var['name'], nonRes_df.columns) for plot_var in PLOT_VARS]],
                    nonRes_aux.loc[cut_mask, ['AUX_hash', 'AUX_eventWeight']]
                ])
            
                if fold_hists[hist_name] is None:
                    fold_hists[hist_name] = copy.deepcopy(pass_cut_df)
                elif len(fold_hists[hist_name]) > 1000: 
                    continue
                else:
                    new_unique_hash = np.setdiff1d(
                        nonRes_aux.loc[cut_mask, 'AUX_hash'],
                        fold_hists[hist_name]['AUX_hash']
                    )
                    if len(new_unique_hash) == 0: continue

                    intersect, comm1, comm2 = np.intersect1d(
                        nonRes_aux.loc[cut_mask, 'AUX_hash'], new_unique_hash, return_indices=True
                    )
                    intersect_bool = np.zeros(np.sum(cut_mask), dtype=bool)
                    for index in comm1: intersect_bool[index] = True
                    
                    fold_hists[hist_name] = pd.concat([fold_hists[hist_name], pass_cut_df.loc[intersect_bool]])
        
        for plot_var in PLOT_VARS:
            tightest_hist = fold_hists[list(fold_hists.keys())[-1]]
            var = match_regex(plot_var['name'], tightest_hist.columns)

            fig, ax = plt.subplots()
            plot_1dhist(
                [hist.loc[:, var].to_numpy() for hist in fold_hists], TRAINING_DIRPATH, PLOT_TYPE, var, 
                weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in fold_hists] 
                if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in fold_hists],
                subplots=(fig, ax), labels=list(fold_hists.keys()), plot_prefix=DISCRIMINATOR, plot_postfix=f"fold{fold_idx}"
            )
            if FIT:
                fit_tightest(tightest_hist.loc[:, var].to_numpy(), var, (fig, ax))
            plt.close()
        
        for hist_name, hist in hists.items():
            if hist is None: hist = copy.deepcopy(fold_hists[hist_name])
            else: hist = pd.concat([hist, fold_hists[hist_name]])
    
    for plot_var in PLOT_VARS:
        tightest_hist = hists[list(hists.keys())[-1]]
        var = match_regex(plot_var['name'], tightest_hist.columns)

        fig, ax = plt.subplots()
        plot_1dhist(
            [hist.loc[:, var].to_numpy() for hist in hists], TRAINING_DIRPATH, PLOT_TYPE, var, 
            weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in hists] 
            if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in hists],
            subplots=(fig, ax), labels=list(hists.keys()), plot_prefix=DISCRIMINATOR
        )
        if FIT:
            fit_tightest(tightest_hist.loc[:, var].to_numpy(), var, (fig, ax))
        plt.close()


if __name__ == "__main__":
    sculpting_check()