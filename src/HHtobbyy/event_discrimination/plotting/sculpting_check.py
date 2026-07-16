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
        hep.cms.text("Preliminary", ax=ax)
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

    excluded_cuts = ['vbfhh_cat1', 'cat2']
    filtered_hist_names = [h for h in hist_names if h not in excluded_cuts]
    
    # PDF 3 - per combo, all cut levels overlaid except cat 2 and vbf hh cat 1
    with PdfPages(os.path.join(output_dirpath, 'sculpting_by_combo_nicer.pdf')) as pdf:
        for combo_name in combo_names:
            fig, axes = plt.subplots(1, len(plot_vars), figsize=(FIG_WIDTH, FIG_HEIGHT))
            if len(plot_vars) == 1: axes = [axes]
            for ax, plot_var in zip(axes, plot_vars):
                counts_list = [hists[combo_name][hist_name][plot_var['name']] for hist_name in filtered_hist_names]
                make_plot(ax, counts_list, filtered_hist_names, plot_var)
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

        # lead_mvaID > -0.7
        lead_mvaID_cut = (-0.7 - 0.7683) / 0.3385 

        # sublead_mvaID > -0.7
        sublead_mvaID_cut = (-0.7 - 0.7270) / 0.3619  

        # 80 < dijet_mass < 190
        dijet_low = (80 - 120.3535) / 31.5187  
        dijet_high = (190 - 120.3535) / 31.5187 
        
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

                    # these are the HiggsDNA presels

                    presel_mask = np.ones(len(df_combo), dtype=bool)
                    presel_mask = presel_mask & (df_combo['lead_mvaID'] > lead_mvaID_cut)
                    presel_mask = presel_mask & (df_combo['sublead_mvaID'] > sublead_mvaID_cut)
                    presel_mask = presel_mask & (df_combo['nonResReg_vbfpair_dijet_mass_DNNreg'] > dijet_low)
                    presel_mask = presel_mask & (df_combo['nonResReg_vbfpair_dijet_mass_DNNreg'] < dijet_high)
                    df_combo = df_combo.loc[presel_mask].reset_index(drop=True)
                    preds = preds[presel_mask]

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

if __name__ == "__main__":
    sculpting_check()