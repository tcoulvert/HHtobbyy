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
import scipy as scp
import matplotlib.pyplot as plt

# HEP packages
import mplhep as hep
import hist
from cycler import cycler

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
from HHtobbyy.event_discrimination.plotting.plot_hists import plot_1dhist
from HHtobbyy.event_discrimination.plotting.plotting_utils import plot_filepath, make_plot_dirpath
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

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def resample_from_var(
    arr, weight, n_events, min_value=None, bins=100, categorical: bool=False
):
    resample_rng = np.random.default_rng(seed=SEED)

    if not categorical:
        np_hist, bin_edges = np.histogram(arr, bins=bins, weights=weight, density=True)
        np_hist /= np.sum(np_hist)

        bin_choices = resample_rng.choice(np.arange(len(np_hist)), size=n_events, p=np_hist)

        value_choices = (bin_edges[bin_choices+1] - bin_edges[bin_choices]) * resample_rng.random(size=n_events) + bin_edges[bin_choices]
    else:
        # value_choices = resample_rng.choice(np.unique(arr), size=n_events, replace=True, p=[np.sum(weight[arr == unique_val]) / np.sum(weight) for unique_val in np.unique(arr)])
        value_choices = np.ones(n_events)

    if min_value is None or np.all(value_choices > min_value):
        return value_choices
    else:  # this is not really correct, just an approximation to make the code faster
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
    if 'AUX_resampled' in var.columns:
        new_rows['AUX_resampled'] = np.ones(new_rows.shape[0], dtype=bool)
    return pd.concat([var, new_rows], ignore_index=True)


def exp_plus_gauss(x, A, tau, B, sigma, C):
    exp = A * np.exp(-x * tau)
    gauss = B * np.exp(-0.5 * ((x - 125) / sigma)**2)
    return exp + gauss + C

def fit_tightest(np_arr, plot_var, subplots, plot_type: str, training_dirpath: str, plot_prefix: str=None, plot_postfix: str=None, ):

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

    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)
    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()


def sculpting_check():
    get_booster = get_model_func(TRAINING_DIRPATH)

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)

    hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}

    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        booster = get_booster(fold_idx)

        minimal_mask = lambda df: np.all(np.array([
            np.logical_and(df.loc[:, match_regex(plot_var['name'], df.columns)] > plot_var['range'][0], df.loc[:, match_regex(plot_var['name'], df.columns)] < plot_var['range'][1])
            for plot_var in PLOT_VARS
        ]).T, axis=1)

        get_resample_filepaths = lambda nonres_samples_key: [
            filepath 
            for class_filepaths in get_filepaths(DATASET_DIRPATH, DATASET, SYST_NAME)(fold_idx).values() 
            for filepath in class_filepaths 
            if match_sample(filepath, [sample[nonres_samples_key] for sample in NONRES_SAMPLES]) is not None
        ]
        nonRes_filepaths = get_resample_filepaths('name')
        nonRes_df, nonRes_aux = pd.concat([get_Dataframe(filepath) for filepath in nonRes_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in nonRes_filepaths])
        nonRes_aux['AUX_resampled'] = np.zeros(nonRes_aux.shape[0], dtype=bool)
        nonRes_df, nonRes_aux = resample_grow_pd(nonRes_df, RESAMPLE), resample_grow_pd(nonRes_aux, RESAMPLE)
        minimal_nonRes_mask = minimal_mask(nonRes_aux)
        nonRes_df, nonRes_aux = nonRes_df.loc[minimal_nonRes_mask].reset_index(drop=True), nonRes_aux.loc[minimal_nonRes_mask].reset_index(drop=True)

        Res_filepaths = get_resample_filepaths('resample_from')
        Res_df, Res_aux = pd.concat([get_Dataframe(filepath) for filepath in Res_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in Res_filepaths])
        minimal_Res_mask = minimal_mask(Res_aux)
        Res_df, Res_aux = Res_df.loc[minimal_Res_mask].reset_index(drop=True), Res_aux.loc[minimal_Res_mask].reset_index(drop=True)

        fold_hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}
        num_iterations = 0
        while True:
            num_iterations += 1

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
                        n_events=np.sum(nonRes_mask), #min_value=nonRes_df.loc[nonRes_mask, "AUX_max_nonbjet_btag"] if "btag" in variable and "WP" not in variable else None
                        categorical=match_sample(variable, ['WP']) is not None,
                    )
                    if match_sample(variable, ['bTagWP']) is not None:  # checking if using WPs, need to set other WPs appropriately
                        init_wp, replace_wps = 'XXT', ['XXT', 'XT', 'T', 'M', 'L']
                        for i, replace_wp in enumerate(replace_wps):
                            if match_sample(variable, [replace_wp]) is not None: 
                                init_wp, replace_wps = replace_wp, replace_wps[i+1:]; break
                        for replace_wp in replace_wps:
                            nonRes_df.loc[nonRes_mask, variable.replace(init_wp, replace_wp)] = np.where(nonRes_df.loc[nonRes_mask, variable] > 0, nonRes_df.loc[nonRes_mask, variable.replace(init_wp, replace_wp)], nonRes_df.loc[nonRes_mask, variable])

            nonRes_dm = get_DMatrix(nonRes_df, nonRes_aux, dataset='test', label=False)
            nonRes_preds = evaluate(booster, nonRes_dm)
            transformed_preds = transform_preds(nonRes_preds)

            for hist_name, cut_dict in SCULPTING_CUTS.items():

                cut_mask = np.ones(np.shape(transformed_preds)[0], dtype=bool)
                for cut_axis, cut_range in cut_dict.items():
                    cut_mask = np.logical_and(
                        cut_mask, 
                        np.logical_and(
                            transformed_preds[:, int(cut_axis)] > cut_range[0],
                            transformed_preds[:, int(cut_axis)] < cut_range[1],
                        )
                    )
                if np.sum(cut_mask) == 0: break

                pass_cut_df = nonRes_aux.loc[cut_mask, [match_regex(plot_var['name'], nonRes_aux.columns) for plot_var in PLOT_VARS]+['AUX_hash', 'AUX_eventWeight']]
            
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
            
            if all(len(fold_hist) > 1000 for fold_hist in fold_hists.values()): break
            else: print('num iterations = ', num_iterations, '\n', '\n'.join([str(len(fold_hist)) for fold_hist in fold_hists.values()]))
        
        for plot_var in PLOT_VARS:
            tightest_hist = fold_hists[list(fold_hists.keys())[-1]]
            var = match_regex(plot_var['name'], tightest_hist.columns)

            fig, ax = plt.subplots()
            plot_1dhist(
                [hist.loc[:, var].to_numpy(dtype=np.float64) for hist in fold_hists.values()], 
                TRAINING_DIRPATH, PLOT_TYPE, var, 
                weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in fold_hists.values()] 
                if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in fold_hists.values()],
                yerr=True, density=DENSITY, logy=LOGY, colors=cmap_petroff10[:len(fold_hists)], _bins=plot_var['bins'], _range=plot_var['range'],
                subplots=(fig, ax), labels=list(fold_hists.keys()), plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_{'weighted' if WEIGHTS else 'unweighted'}_fold{fold_idx}", save_and_close=not FIT
            )
            if FIT:
                fit_tightest(tightest_hist.loc[:, var].to_numpy(), plot_var, (fig, ax), PLOT_TYPE, TRAINING_DIRPATH, plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_fit_fold{fold_idx}")
        
        for hist_name in hists.keys():
            if hists[hist_name] is None: hists[hist_name] = copy.deepcopy(fold_hists[hist_name])
            else: hists[hist_name] = pd.concat([hists[hist_name], fold_hists[hist_name]])
    
    for plot_var in PLOT_VARS:
        tightest_hist = hists[list(hists.keys())[-1]]
        var = match_regex(plot_var['name'], tightest_hist.columns)

        fig, ax = plt.subplots()
        plot_1dhist(
            [hist.loc[:, var].to_numpy(dtype=np.float64) for hist in hists.values()], 
            TRAINING_DIRPATH, PLOT_TYPE, var, 
            weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in hists.values()] 
            if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in hists.values()],
            yerr=True, density=DENSITY, logy=LOGY, colors=cmap_petroff10[:len(hists)], _bins=plot_var['bins'], _range=plot_var['range'],
            subplots=(fig, ax), labels=list(hists.keys()), plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_{'weighted' if WEIGHTS else 'unweighted'}", save_and_close=not FIT
        )
        if FIT:
            fit_tightest(tightest_hist.loc[:, var].to_numpy(), plot_var, (fig, ax), PLOT_TYPE, TRAINING_DIRPATH, plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_fit")


if __name__ == "__main__":
    sculpting_check()