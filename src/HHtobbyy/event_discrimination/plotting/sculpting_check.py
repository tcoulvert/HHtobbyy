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

#time stamping

import datetime


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)

## (below) since its already a package we dont need these added to the sys path plus its using an older directory structure 

# sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
# sys.path.append(os.path.join(GIT_REPO, "training/"))
# sys.path.append(os.path.join(GIT_REPO, "evaluation/"))
    

# Module packages
from HHtobbyy.event_discrimination.plotting.plot_hists import plot_1dhist
from HHtobbyy.event_discrimination.plotting.plotting_utils import plot_filepath, make_plot_dirpath

# removed get_Dataframe, get_DMatrix, get_class_sample_map, get_n_folds, and get_filepaths & some more from here since they no longer exist rip

from HHtobbyy.workspace_utils.retrieval_utils import (
    match_regex,
    match_sample,
    format_class_names,
)
from HHtobbyy.event_discrimination.DFDataset import DFDataset

from HHtobbyy.event_discrimination.models import map_model_to_Model

from HHtobbyy.event_discrimination.evaluation.evaluation_utils import (
    transform_preds_options, transform_preds_func,
)



################################

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")

## (below) obsolete
# parser.add_argument(
#     "training_dirpath",
#     help="Full filepath for trained model files"
# )

# added this to specify DFDataset config file
parser.add_argument(
    "dfdataset_config",
    help="Full filepath for DFDataset config file"
)

# added this to choose model 
parser.add_argument(
    "model",
    choices=["MLP", "XGBoostBDT"],
    help="What model to use (eg. MLP or XGBoostBDT)"
)

# added this to specify model config file
parser.add_argument(
    "model_config",
    help="Full filepath for model config file"
)
parser.add_argument(
    "sculpting_cuts",
    help="JSON file that defines the discriminator(s) to use, the cuts to apply, and other configurations for the sculpting check"
)

## (below) obsolete
# parser.add_argument(
#     "--dataset_dirpath", 
#     default=None,
#     help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_dirpath.txt` file"
# )

# parser.add_argument(
#     "--rebin",
#     type=int,
#     default=1,
#     help="Number of bins to merge when plotting"
# )

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
    default=10,  # 1 to starts
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

# jupyter notebook error
args = parser.parse_args()

# (below) obsolete since we're now using DFDataset config file to get dataset_dirpath
# TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
# if args.dataset_dirpath is None:
#     DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
# else:
#     DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')

DFDATASET_CONFIG = args.dfdataset_config
MODEL_CONFIG = args.model_config
MODEL_TYPE = args.model
DATASET = args.dataset # might not need later?
WEIGHTS = args.weights
DENSITY = args.density
LOGY = args.logy
RESAMPLE = args.resample # also might not need later?
SYST_NAME = args.syst_name
SEED = args.seed
FIT = args.fit
# REBIN = args.rebin

dfdataset = DFDataset(args.dfdataset_config)
columns_to_load = list(dfdataset.all_vars_map.keys())
model = map_model_to_Model(args.model)(dfdataset, args.model_config)
CLASS_NAMES = list(dfdataset.class_sample_map.keys())
PLOT_TYPE = "resample_sculpting"

with open(args.sculpting_cuts, 'r') as f:
    SCULPTING_CUTS = json.load(f)
DISCRIMINATOR = SCULPTING_CUTS.pop('discriminator')
assert DISCRIMINATOR in transform_preds_options(), f"Trying to use a discriminator ({DISCRIMINATOR}) that isn't implemented in evaluation_utils. Use one of {transform_preds_options()} or implement your own"
NONRES_SAMPLES = SCULPTING_CUTS.pop('nonRes_samples')
RESAMPLE_VARS = SCULPTING_CUTS.pop('resample_vars')
RESAMPLE_COMBOS = SCULPTING_CUTS.pop('resample_combos')
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

    # else:
    #     # value_choices = resample_rng.choice(np.unique(arr), size=n_events, replace=True, p=[np.sum(weight[arr == unique_val]) / np.sum(weight) for unique_val in np.unique(arr)])
    #     value_choices = np.ones(n_events)

    # if min_value is None or np.all(value_choices > min_value):
    #     return value_choices
    # else:  # this is not really correct, just an approximation to make the code faster
    #     bad_choices_bool = value_choices <= min_value

    #     largest_min_value = np.max(min_value[bad_choices_bool])

    #     np_hist, bin_edges = np.histogram(arr[arr > largest_min_value], bins=bins, weights=arr[arr > largest_min_value], density=True)
    #     np_hist /= np.sum(np_hist)

    #     bin_choices = resample_rng.choice(np.arange(len(np_hist)), size=np.sum(bad_choices_bool), p=np_hist)
        
    #     value_choices[bad_choices_bool] = (bin_edges[bin_choices+1] - bin_edges[bin_choices]) * resample_rng.random(size=np.sum(bad_choices_bool)) + bin_edges[bin_choices]
        
        return value_choices


def exp_plus_gauss(x, A, tau, B, sigma, C):
    exp = A * np.exp(-x * tau)
    gauss = B * np.exp(-0.5 * ((x - 125) / sigma)**2)
    return exp + gauss + C

def fit_tightest(np_arr, plot_var, subplots, plot_type: str, base_dirpath: str, plot_prefix: str='', plot_postfix: str=''):

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

    plot_dirpath = make_plot_dirpath(base_dirpath, plot_type)
    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()

resample_rng = np.random.default_rng(seed=SEED)

def plot_sculpting(npy_filepath, output_dirpath='.'):
    data = np.load(npy_filepath, allow_pickle=True).item()
    hists = data['hists']
    plot_vars = data['plot_vars']
    combo_names = list(hists.keys())
    hist_names = list(list(hists.values())[0].keys())
    FIG_WIDTH = 8 * len(plot_vars)
    FIG_HEIGHT = 7

    from matplotlib.backends.backend_pdf import PdfPages

    def make_axes(plot_var):
        rebin = 4 if 'HH' in plot_var['name'] else plot_var.get('rebin', 1)
        n_bins = plot_var['bins'] // rebin
        plot_bin_edges = np.linspace(plot_var['range'][0], plot_var['range'][1], n_bins + 1)
        bin_centers = (plot_bin_edges[:-1] + plot_bin_edges[1:]) / 2
        bin_width = plot_bin_edges[1] - plot_bin_edges[0]
        return rebin, n_bins, plot_bin_edges, bin_centers, bin_width

    def make_plot(ax, counts_list, labels, plot_var):
        rebin, n_bins, plot_bin_edges, bin_centers, bin_width = make_axes(plot_var)
        for counts, label, color in zip(counts_list, labels, cmap_petroff10[:len(counts_list)]):
            rebinned = np.add.reduceat(counts, np.arange(0, plot_var['bins'], rebin))
            total = np.sum(rebinned)
            if total == 0: continue
            density = rebinned / (total * bin_width)
            density_err = np.sqrt(rebinned) / (total * bin_width)
            ax.stairs(density, plot_bin_edges, label=label, color=color)
            ax.errorbar(bin_centers, density, yerr=density_err, fmt='none', color=color)
        hep.cms.text("Simulation Preliminary", ax=ax)
        ax.legend(loc='upper right', fontsize=14)
        ax.set_xlabel(f"{plot_var['name']} [GeV] / {bin_width:.1f} GeV")
        ax.set_ylabel('Density')


    # PDF 1 - per cut level, all combos overlaid
    with PdfPages(os.path.join(output_dirpath, 'sculpting_by_cut.pdf')) as pdf:
        for hist_name in hist_names:
            fig, axes = plt.subplots(1, len(plot_vars), figsize=(FIG_WIDTH, FIG_HEIGHT))
            if len(plot_vars) == 1: axes = [axes]
            for ax, plot_var in zip(axes, plot_vars):
                counts_list = [hists[combo_name][hist_name][plot_var['name']] for combo_name in combo_names]
                make_plot(ax, counts_list, combo_names, plot_var)
            fig.suptitle(hist_name, fontsize=16, y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    # PDF 2 - per combo, all cut levels overlaid
    with PdfPages(os.path.join(output_dirpath, 'sculpting_by_combo.pdf')) as pdf:
        for combo_name in combo_names:
            fig, axes = plt.subplots(1, len(plot_vars), figsize=(FIG_WIDTH, FIG_HEIGHT))
            if len(plot_vars) == 1: axes = [axes]
            for ax, plot_var in zip(axes, plot_vars):
                counts_list = [hists[combo_name][hist_name][plot_var['name']] for hist_name in hist_names]
                make_plot(ax, counts_list, hist_names, plot_var)
            fig.suptitle(combo_name, fontsize=16, y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def sculpting_check():

    run_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dirpath = os.path.join('/eos/uscms/store/group/lpcdihiggsboost/nparekh/sculpting_checks', run_time)
    os.makedirs(run_dirpath, exist_ok=True)

    hists = {combo['name']: {hist_name: {plot_var['name']: np.zeros(plot_var['bins']) for plot_var in PLOT_VARS} for hist_name in SCULPTING_CUTS.keys()} for combo in RESAMPLE_COMBOS}

    formatted_class_names = format_class_names(CLASS_NAMES)
    
    for fold in range(dfdataset.n_folds):
        
        fold_hists = {combo['name']: {hist_name: {plot_var['name']: np.zeros(plot_var['bins']) for plot_var in PLOT_VARS} for hist_name in SCULPTING_CUTS.keys()} for combo in RESAMPLE_COMBOS}


        resample_hists = {var['name']: np.zeros(var['bins']) for var in RESAMPLE_VARS if 'value' not in var}

        signal_filepaths = dfdataset.get_traintest_filepaths(fold, dataset="test", syst_name=SYST_NAME)['ggF HH']

        print(f"Collecting signal photon ID for fold {fold}...")

        for filepath in signal_filepaths:

            print(f"    Processing signal file: {filepath}")
            batch_ctr = 0 

            for batch in dfdataset.get_df_iter(filepath, batch_size=131_072, filter=dfdataset.presel_filter):

                print(f"        Processing signal batch: {batch_ctr}")
                batch_ctr += 1

                df = batch.to_pandas()

                for var in RESAMPLE_VARS:
                    if 'value' in var: continue
                    vals = df[var['name']].to_numpy()
                    vals = vals[vals != dfdataset.refill_value]
                    resample_hists[var['name']] += np.histogram(vals, bins=var['bins'], range=var['range'])[0]
        
        for var in RESAMPLE_VARS:
            if 'value' in var: continue
            resample_hists[var['name']] /= np.sum(resample_hists[var['name']])

        bkg_filepaths = [
            fp for fp in dfdataset.get_traintest_filepaths(fold, dataset="test", syst_name=SYST_NAME)['nonRes']
            if match_sample(fp, [sample['name'] for sample in NONRES_SAMPLES]) is not None
        ]

        print(f"Signal collection done.")
        
        print(f"Processing background for fold {fold}...")

        ckpt_path = model.modelconfig.get_ckpt_path(fold)
        mlp_model, trainer = model.load_model_and_trainer(ckpt_path=ckpt_path, eval=True)

        for filepath in bkg_filepaths:

            print(f"    Processing bkg file: {filepath}")
            batch_ctr = 0 
            for batch in dfdataset.get_df_iter(filepath, filter=dfdataset.presel_filter):

                print(f"        Processing bkg batch: {batch_ctr}")
                batch_ctr += 1

                df = batch.to_pandas()

                for combo in RESAMPLE_COMBOS:
                    df_combo = df.copy()
                    for var_name in combo['vars']:
                        var = next(v for v in RESAMPLE_VARS if v['name'] == var_name)
                        if 'value' in var:
                            cols = [col for col in df_combo.columns if match_regex(var['name'], [col]) is not None]
                            for col in cols:
                                df_combo[col] = var['value']
                        else:
                            bin_edges = np.linspace(var['range'][0], var['range'][1], var['bins'] + 1)
                            bin_choices = resample_rng.choice(np.arange(var['bins']), size=len(df_combo), p=resample_hists[var['name']])
                            bin_width = (var['range'][1] - var['range'][0]) / var['bins']
                            df_combo[var['name']] = bin_width * resample_rng.random(size=len(df_combo)) + bin_edges[bin_choices]

                    data = model.modeldataset.get_data(df_combo, dfdataset.event_weight_var)
                    predictions = trainer.predict(mlp_model, data)
                    preds = np.concatenate([prediction.numpy(force=True) for prediction in predictions])

                    for hist_name, cut_dict in SCULPTING_CUTS.items():
                        cut_mask = np.ones(len(preds), dtype=bool)
                        for column, cut in cut_dict.items():
                            class_name = column.replace('AUX_D', '')
                            class_idx = formatted_class_names.index(class_name)
                            if "HH" in class_name:
                                cut_mask = cut_mask & (preds[:, class_idx] > cut)
                            else:
                                cut_mask = cut_mask & (preds[:, class_idx] < cut)

                        for plot_var in PLOT_VARS:
                            col = match_regex(plot_var['name'], df_combo.columns)
                            if col is None: continue
                            vals = df_combo.loc[cut_mask, col].values
                            vals = vals[vals != dfdataset.refill_value]
                            fold_hists[combo['name']][hist_name][plot_var['name']] += np.histogram(vals, bins=plot_var['bins'], range=plot_var['range'])[0]

        print(f"Fold {fold} done.")

        for combo in RESAMPLE_COMBOS:
            for hist_name in hists[combo['name']].keys():
                for plot_var in PLOT_VARS:
                    hists[combo['name']][hist_name][plot_var['name']] += fold_hists[combo['name']][hist_name][plot_var['name']]

    print("Plotting...")

    np.save(os.path.join(run_dirpath, 'sculpting_hists.npy'), {
        'hists': hists,
        'resample_vars': RESAMPLE_VARS,
        'resample_combos': RESAMPLE_COMBOS,
        'sculpting_cuts': SCULPTING_CUTS,
        'plot_vars': PLOT_VARS,
        'discriminator': DISCRIMINATOR,
    })

    plot_sculpting(os.path.join(run_dirpath, 'sculpting_hists.npy'), run_dirpath)


# #Thomas code
# def sculpting_check(nonres_bkg_files: list, signal_files: list):
#     get_booster = get_model_func(TRAINING_DIRPATH)

#     transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, DISCRIMINATOR)

#     hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}

#     dfdataset = <instantiate_dfdataset>

#     signal_dfs = pd.concat([pd.read_parquet(signal_file, columns=[f'{}mass', f'{dfdataset.aux_var_prefix}mass']) for signal_file in signal_files])

#     # wrap this in batching decorator
#     def make_signal_distributions(df: pd.DataFrame, storage, var_cols, etc):
#         for var_col1 in var_cols:
#             var = df[var1_col]
#             storage = np.concatenate([storage, var])  # storage is numpy array of values
#             # OR
#             storage = np.concatenate([storage, np.hist(var, bins=n_bins, range=(start, stop))])  # storage is numpy histogram of bins


#     var_arr = make_signal_distributions
#     hist_var_arr = np.hist(var_arr)

#     # repeat for every variable
#     def resample_bkg(df: pd.DataFrame, signal_hists: list[np.hists], var_cols, mass_values, cuts, etc):
#         for signal_hist, var_col1 in zip(signal_hists, var_cols):
#             df[var1_col] = resample_variable(df[var1_col], signal_hist)
#         data = model.modeldataset.get_data(df)
#         predictions = model.get_predictions(data)

#         mass_values = np.concatenate([mass_values, prediction[predictions > cuts]])

#     bkg_mass_values = resample_bkg
#     hist_mass = np.hist(bkg_mass_values)

    # plot bkg hist_mass

    # for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
    #     booster = get_booster(fold_idx)

    #     minimal_mask = lambda df: np.all(np.array([
    #         np.logical_and(df.loc[:, match_regex(plot_var['name'], df.columns)] > plot_var['range'][0], df.loc[:, match_regex(plot_var['name'], df.columns)] < plot_var['range'][1])
    #         for plot_var in PLOT_VARS
    #     ]).T, axis=1)

    #     get_resample_filepaths = lambda nonres_samples_key: [
    #         filepath 
    #         for class_filepaths in get_filepaths(DATASET_DIRPATH, DATASET, SYST_NAME)(fold_idx).values() 
    #         for filepath in class_filepaths 
    #         if match_sample(filepath, [sample[nonres_samples_key] for sample in NONRES_SAMPLES]) is not None
    #     ]
    #     nonRes_filepaths = get_resample_filepaths('name')
    #     nonRes_df, nonRes_aux = pd.concat([get_Dataframe(filepath) for filepath in nonRes_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in nonRes_filepaths])
    #     nonRes_aux['AUX_resampled'] = np.zeros(nonRes_aux.shape[0], dtype=bool)
    #     nonRes_df, nonRes_aux = resample_grow_pd(nonRes_df, RESAMPLE), resample_grow_pd(nonRes_aux, RESAMPLE)
    #     minimal_nonRes_mask = minimal_mask(nonRes_aux)
    #     nonRes_df, nonRes_aux = nonRes_df.loc[minimal_nonRes_mask].reset_index(drop=True), nonRes_aux.loc[minimal_nonRes_mask].reset_index(drop=True)

    #     Res_filepaths = get_resample_filepaths('resample_from')
    #     Res_df, Res_aux = pd.concat([get_Dataframe(filepath) for filepath in Res_filepaths]), pd.concat([get_Dataframe(filepath, aux=True) for filepath in Res_filepaths])
    #     minimal_Res_mask = minimal_mask(Res_aux)
    #     Res_df, Res_aux = Res_df.loc[minimal_Res_mask].reset_index(drop=True), Res_aux.loc[minimal_Res_mask].reset_index(drop=True)

    #     fold_hists = {hist_name: None for hist_name in SCULPTING_CUTS.keys()}

    #     for nonRes_sample in NONRES_SAMPLES:
    #         nonRes_mask = np.logical_and(
    #             nonRes_aux.loc[:, "AUX_sample_name"] == nonRes_sample["name"],
    #             nonRes_aux.loc[:, "AUX_resampled"]
    #         )
    #         Res_mask = (Res_aux.loc[:, "AUX_sample_name"] == nonRes_sample["resample_from"])

    #         for resample_var in nonRes_sample["resample_vars"]:
    #             variable = match_regex(resample_var, nonRes_df.columns)
    #             assert variable is not None, f"Variable with regex string {resample_var} does not exist in Dataframe, check if the regex string is correct"
    #             nonRes_df.loc[nonRes_mask, variable] = resample_from_var(
    #                 Res_df.loc[Res_mask, variable], weight=Res_aux.loc[Res_mask, "AUX_eventWeight"] if WEIGHTS else np.ones(np.sum(Res_mask)),
    #                 n_events=np.sum(nonRes_mask), #min_value=nonRes_df.loc[nonRes_mask, "AUX_max_nonbjet_btag"] if "btag" in variable and "WP" not in variable else None
    #                 categorical=match_sample(variable, ['WP']) is not None,
    #             )
    #             if match_sample(variable, ['bTagWP']) is not None:  # checking if using WPs, need to set other WPs appropriately
    #                 init_wp, replace_wps = 'XXT', ['XXT', 'XT', 'T', 'M', 'L']
    #                 for i, replace_wp in enumerate(replace_wps):
    #                     if match_sample(variable, [replace_wp]) is not None: 
    #                         init_wp, replace_wps = replace_wp, replace_wps[i+1:]; break
    #                 for replace_wp in replace_wps:
    #                     nonRes_df.loc[nonRes_mask, variable.replace(init_wp, replace_wp)] = np.where(nonRes_df.loc[nonRes_mask, variable] > 0, nonRes_df.loc[nonRes_mask, variable.replace(init_wp, replace_wp)], nonRes_df.loc[nonRes_mask, variable])

    #         nonRes_dm = get_DMatrix(nonRes_df, nonRes_aux, dataset='test', label=False)  # model.get_data(df, etc)
    #         nonRes_preds = evaluate(booster, nonRes_dm)  # model.predict_data(data, etc)
    #         transformed_preds = transform_preds(nonRes_preds)

    #         for hist_name, cut_dict in SCULPTING_CUTS.items():

    #             cut_mask = np.ones(np.shape(transformed_preds)[0], dtype=bool)
    #             for cut_axis, cut_range in cut_dict.items():
    #                 cut_mask = np.logical_and(
    #                     cut_mask, 
    #                     np.logical_and(
    #                         transformed_preds[:, int(cut_axis)] > cut_range[0],
    #                         transformed_preds[:, int(cut_axis)] < cut_range[1],
    #                     )
    #                 )
    #             if np.sum(cut_mask) == 0: break

    #             pass_cut_df = nonRes_aux.loc[cut_mask, [match_regex(plot_var['name'], nonRes_aux.columns) for plot_var in PLOT_VARS]+['AUX_hash', 'AUX_eventWeight']]
            
    #             if fold_hists[hist_name] is None:
    #                 fold_hists[hist_name] = copy.deepcopy(pass_cut_df)
    #             elif len(fold_hists[hist_name]) > 1000: 
    #                 continue
    #             else:
    #                 new_unique_hash = np.setdiff1d(
    #                     nonRes_aux.loc[cut_mask, 'AUX_hash'],
    #                     fold_hists[hist_name]['AUX_hash']
    #                 )
    #                 if len(new_unique_hash) == 0: continue

    #                 intersect, comm1, comm2 = np.intersect1d(
    #                     nonRes_aux.loc[cut_mask, 'AUX_hash'], new_unique_hash, return_indices=True
    #                 )
    #                 intersect_bool = np.zeros(np.sum(cut_mask), dtype=bool)
    #                 for index in comm1: intersect_bool[index] = True
                    
    #                 fold_hists[hist_name] = pd.concat([fold_hists[hist_name], pass_cut_df.loc[intersect_bool]])
            
    #         if all(len(fold_hist) > 1000 for fold_hist in fold_hists.values()): break
    #         else: print('num iterations = ', num_iterations, '\n', '\n'.join([str(len(fold_hist)) for fold_hist in fold_hists.values()]))
        

# #Thomas code
#         for plot_var in PLOT_VARS:
#             tightest_hist = fold_hists[list(fold_hists.keys())[-1]]
#             var = match_regex(plot_var['name'], tightest_hist.columns)

#             fig, ax = plt.subplots()
#             plot_1dhist(
#                 [hist.loc[:, var].to_numpy(dtype=np.float64) for hist in fold_hists.values()], 
#                 TRAINING_DIRPATH, PLOT_TYPE, var, 
#                 weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in fold_hists.values()] 
#                 if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in fold_hists.values()],
#                 yerr=True, density=DENSITY, logy=LOGY, colors=cmap_petroff10[:len(fold_hists)], _bins=plot_var['bins'], _range=plot_var['range'],
#                 subplots=(fig, ax), labels=list(fold_hists.keys()), plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_{'weighted' if WEIGHTS else 'unweighted'}_fold{fold_idx}", save_and_close=not FIT
#             )
#             if FIT:
#                 fit_tightest(tightest_hist.loc[:, var].to_numpy(), plot_var, (fig, ax), PLOT_TYPE, TRAINING_DIRPATH, plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_fit_fold{fold_idx}")
        
#         for hist_name in hists.keys():
#             if hists[hist_name] is None: hists[hist_name] = copy.deepcopy(fold_hists[hist_name])
#             else: hists[hist_name] = pd.concat([hists[hist_name], fold_hists[hist_name]])
    
#     for plot_var in PLOT_VARS:
#         tightest_hist = hists[list(hists.keys())[-1]]
#         var = match_regex(plot_var['name'], tightest_hist.columns)

#         fig, ax = plt.subplots()
#         plot_1dhist(
#             [hist.loc[:, var].to_numpy(dtype=np.float64) for hist in hists.values()], 
#             TRAINING_DIRPATH, PLOT_TYPE, var, 
#             weights=[hist.loc[:, 'AUX_eventWeight'].to_numpy() for hist in hists.values()] 
#             if WEIGHTS else [np.ones_like(hist.loc[:, 'AUX_eventWeight']) for hist in hists.values()],
#             yerr=True, density=DENSITY, logy=LOGY, colors=cmap_petroff10[:len(hists)], _bins=plot_var['bins'], _range=plot_var['range'],
#             subplots=(fig, ax), labels=list(hists.keys()), plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_{'weighted' if WEIGHTS else 'unweighted'}", save_and_close=not FIT
#         )
#         if FIT:
#             fit_tightest(tightest_hist.loc[:, var].to_numpy(), plot_var, (fig, ax), PLOT_TYPE, TRAINING_DIRPATH, plot_prefix=DISCRIMINATOR, plot_postfix=f"{''.join(plot_var['name'].split('*'))}_fit")


if __name__ == "__main__":
    sculpting_check()


