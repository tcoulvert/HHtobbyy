# Stdlib packages
import argparse
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
from plotting_utils import (
    make_plot_dirpath, plot_filepath, 
    make_plot_data, combine_prepostfix
)
from retrieval_utils import (
    get_class_sample_map, get_n_folds, match_sample,
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

def resample_grow_np(var, bool_arr, n_duplicates_per_event):
    new_rows_shape = tuple([n_duplicates_per_event]+[1 for _ in range(1, len(np.shape(var)))])
    new_rows = np.tile(
        var[bool_arr],
        new_rows_shape
    )
    return np.concatenate([var, new_rows])

def resample_grow_pd(var, n_duplicates_per_event):
    new_rows = pd.DataFrame(
        np.tile(var.to_numpy(), (n_duplicates_per_event, 1)), columns=var.columns
    )
    new_rows['AUX_resampled'] = np.ones_like(new_rows['AUX_hash'].to_numpy(), dtype=bool)
    return pd.concat([var, new_rows], ignore_index=True)
        



def sculpting_check():
    make_plot_dirpath(TRAINING_DIRPATH, PLOT_TYPE)

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

        while True:

            for nonRes_sample in NONRES_SAMPLES:
                nonRes_mask = np.logical_and(
                    nonRes_aux.loc[:, "AUX_sample_name"] == nonRes_sample["name"],
                    nonRes_aux.loc[:, "AUX_resampled"]
                )
                Res_mask = (Res_aux.loc[:, "AUX_sample_name"] == nonRes_sample["resample_from"])

                for resample_var in nonRes_sample["resample_vars"]:
                    try:
                        variable = [field for field in sorted(nonRes_df.fields, key=len) if match_sample(field, {resample_var}) is not None][0]
                    except IndexError as e:
                        logger.error(f"Variable with regex string {resample_var} does not exist in Dataframe, check if the regex string is correct")
                        raise e
                    nonRes_df.loc[nonRes_mask, variable] = resample_from_var(
                        Res_df.loc[Res_mask, variable], weight=Res_aux.loc[Res_mask, "AUX_eventWeight"] if WEIGHTS else np.ones(np.sum(Res_mask)),
                        n_events=np.sum(nonRes_mask), min_value=nonRes_df.loc[nonRes_mask, "AUX_max_nonbjet_btag"] if "btag" in variable and "WP" not in variable else None
                    )

            nonRes_dm = get_DMatrix(nonRes_df, nonRes_aux, dataset='test', label=False)
            nonRes_preds = evaluate(booster, nonRes_dm)
            transformed_preds = transform_preds(nonRes_preds)

            for hist_name, cut_dict in SCULPTING_CUTS.items():
                if hists[hist_name] is not None and len(hists[hist_name]) > 1000: continue

                



    for fold_idx in range(len(bdt_train_dict)):
        booster = xgb.Booster(param)
        booster.load_model(os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_BDT_fold{fold_idx}.model'))

        nonres_bool = (data_test_aux_dict[f"fold_{fold_idx}"].loc[:, 'sample_name'] == "GGJets") | (data_test_aux_dict[f"fold_{fold_idx}"].loc[:, 'sample_name'] == "GJetPt20To40") | (data_test_aux_dict[f"fold_{fold_idx}"].loc[:, 'sample_name'] == "GJetPt40")

        data_hlf_test = resample_grow_np(data_hlf_test_dict[f"fold_{fold_idx}"], nonres_bool, resample_)
        data_test_aux = resample_grow_pd(data_test_aux_dict[f"fold_{fold_idx}"], nonres_bool, resample_)
        weight_test = resample_grow_np(weight_test_dict[f"fold_{fold_idx}"], nonres_bool, resample_)
        weights_plot = resample_grow_np(weights_plot_test[f"fold_{fold_idx}"], nonres_bool, resample_)
        xgb_label_test = resample_grow_np(xgb_label_test_dict[f"fold_{fold_idx}"], nonres_bool, resample_)

        gg_bool = (data_test_aux.loc[:, 'sample_name'] == "GGJets")
        tth_bool = (data_test_aux.loc[:, 'sample_name'] == "ttHToGG")
        gj_bool = (data_test_aux.loc[:, 'sample_name'] == "GJetPt20To40") | (data_test_aux.loc[:, 'sample_name'] == "GJetPt40")
        nonres_bool = (data_test_aux.loc[:, 'sample_name'] == "GGJets") | (data_test_aux.loc[:, 'sample_name'] == "GJetPt20To40") | (data_test_aux.loc[:, 'sample_name'] == "GJetPt40")


        for _ in range(RESAMPLE // resample_):

            for particle_type in ['lead', 'sublead']:

                gg_mvaID = data_hlf_test[
                    gg_bool, 
                    hlf_vars_columns_dict[f"fold_{fold_idx}"][f"{particle_type}_mvaID"]
                ]
                data_hlf_test[
                    gj_bool, 
                    hlf_vars_columns_dict[f"fold_{fold_idx}"][f"{particle_type}_mvaID"]
                ] = resample_from_var(
                    gg_mvaID, 
                    weights_plot[gg_bool],
                    np.sum(gj_bool),
                    bins=190
                )

                tth_pNetB = data_hlf_test[
                    tth_bool, 
                    hlf_vars_columns_dict[f"fold_{fold_idx}"][f"{particle_type}_bjet_btagPNetB"]
                ]
                data_hlf_test[
                    nonres_bool, 
                    hlf_vars_columns_dict[f"fold_{fold_idx}"][f"{particle_type}_bjet_btagPNetB"]
                ] = resample_from_var(
                    tth_pNetB, 
                    np.abs(weights_plot[tth_bool]),
                    np.sum(nonres_bool),
                    bins=100,
                    min_value=data_test_aux.loc[nonres_bool, "max_nonbjet_btag"].to_numpy()
                )

            nonres_ggf_preds = booster.predict(
                xgb.DMatrix(
                    data=data_hlf_test[nonres_bool], label=xgb_label_test[nonres_bool], 
                    weight=np.abs(weight_test)[nonres_bool],
                    missing=-999.0, feature_names=list(hlf_vars_columns_dict[f"fold_{fold_idx}"])
                ), 
                iteration_range=(0, booster.best_iteration+1)
            )[:, 0]
            
            for score_cut in score_cuts:
                if (
                    len(BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_vars[0]]) > 0 
                    and len(np.concatenate(BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_vars[0]])) >= 1000
                ):
                    continue

                new_unique_eventNumber = np.setdiff1d(
                    data_test_aux.loc[nonres_bool, "event"].to_numpy()[nonres_ggf_preds > score_cut],
                    BDT_perf_resample[fold_idx][f'preds{score_cut}']["event"]
                )
                
                if len(new_unique_eventNumber) > 0:
                    BDT_perf_resample[fold_idx][f'preds{score_cut}']["event"].extend(new_unique_eventNumber.tolist())

                    intersect, comm1, comm2 = np.intersect1d(
                        data_test_aux.loc[nonres_bool, "event"].to_numpy()[nonres_ggf_preds > score_cut],
                        new_unique_eventNumber,
                        return_indices=True
                    )

                    intersect_bool = np.zeros_like(
                        data_test_aux.loc[nonres_bool, "event"].to_numpy()[nonres_ggf_preds > score_cut], 
                        dtype=bool
                    )
                    for index in comm1:
                        intersect_bool[index] = True
                    
                    for var_idx, plot_var in enumerate(plot_vars):
                        BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_var].append(
                            data_test_aux.loc[nonres_bool, plot_var].to_numpy()[nonres_ggf_preds > score_cut][intersect_bool]
                        )
                    
        for var_idx, plot_var in enumerate(plot_vars):

            plot_dirpath_ = os.path.join(plot_dirpath, plot_var)
            if not os.path.exists(plot_dirpath_):
                os.makedirs(plot_dirpath_)

            for score_cut in score_cuts:
                with open(f"resampled_{plot_var}_at{str(score_cut).replace('.', 'p')}_fold{fold_idx}.npy", "wb") as f:
                    np.save(f, BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_var])
            
            test_hists = [hist.Hist(VARIABLES[plot_var]).fill(var=np.concatenate(BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_var])) for score_cut in score_cuts]
            make_input_plot(
                plot_dirpath_, plot_var,
                test_hists, 
                fold_idx=fold_idx, labels=label_arr, 
                plot_prefix='test_non-res_scoreCut_'
            )

    for var_idx, plot_var in enumerate(plot_vars):

        plot_dirpath_ = os.path.join(plot_dirpath, plot_var)
        if not os.path.exists(plot_dirpath_):
            os.makedirs(plot_dirpath_)

        concat_samples = {
            score_cut: np.concatenate(
                [np.concatenate(BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_var]) for fold_idx in range(len(BDT_perf_resample))]
            ) for score_cut in score_cuts
        }
        for score_cut in score_cuts:
            with open(f"resampled_{plot_var}_at{str(score_cut).replace('.', 'p')}_all.npy", "wb") as f:
                np.save(f, concat_samples[score_cut])

        test_hists = [hist.Hist(VARIABLES[plot_var]).fill(
            var=np.concatenate(
                [np.concatenate(BDT_perf_resample[fold_idx][f'preds{score_cut}'][plot_var]) for fold_idx in range(len(BDT_perf_resample))]
            )
        ) for score_cut in score_cuts]
        make_input_plot(
            plot_dirpath_, plot_var,
            test_hists, 
            fold_idx=None, labels=label_arr, 
            plot_prefix='test_non-res_scoreCut_'
        )



if __name__ == "__main__":
    pass









files = [
    'resampled_mass_at0p0_all.npy',    'resampled_mass_at0p9955_all.npy',
    'resampled_mass_at0p0_fold0.npy',  'resampled_mass_at0p9955_fold0.npy',
    'resampled_mass_at0p0_fold1.npy',  'resampled_mass_at0p9955_fold1.npy',
    'resampled_mass_at0p0_fold2.npy',  'resampled_mass_at0p9955_fold2.npy',
    'resampled_mass_at0p0_fold3.npy',  'resampled_mass_at0p9955_fold3.npy',
    'resampled_mass_at0p0_fold4.npy',  'resampled_mass_at0p9955_fold4.npy',
    'resampled_mass_at0p7_all.npy',    'resampled_mass_at0p99_all.npy',
    'resampled_mass_at0p7_fold0.npy',  'resampled_mass_at0p99_fold0.npy',
    'resampled_mass_at0p7_fold1.npy',  'resampled_mass_at0p99_fold1.npy',
    'resampled_mass_at0p7_fold2.npy',  'resampled_mass_at0p99_fold2.npy',
    'resampled_mass_at0p7_fold3.npy',  'resampled_mass_at0p99_fold3.npy',
    'resampled_mass_at0p7_fold4.npy',  'resampled_mass_at0p99_fold4.npy',
]

def exp_plus_gauss(x, A, tau, B, sigma, C):
    exp = A * np.exp(-x * tau)
    gauss = B * np.exp(-0.5 * ((x - 125) / sigma)**2)
    return exp + gauss + C

for file in files:
    if 'all' not in file: continue

    with open(file, "rb") as f:
        np_arr = np.load(f, allow_pickle=True)

    np_hist, bin_edges = np.histogram(np_arr, bins=N_BINS, range=RANGE, density=True)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(np_hist))]

    popt, pcov = scp.optimize.curve_fit(exp_plus_gauss, bin_centers, np_hist, p0=[10, 1/70, 1, 2, 0])
    perr = np.sqrt(np.diag(pcov))

    print('-'*60)
    print(file)
    for i, param in enumerate(['Exp Amp', 'Exp tau', 'Gauss Amp', 'Gauss sigma', 'Y-intercept']):
        print(f"{param} = {popt[i]:.4f}±{perr[i]:.4f}")

    x_trial = np.linspace(100, 180, 1000)
    y_fit = exp_plus_gauss(x_trial, *popt)
    plt.hist(np_arr, bins=N_BINS, range=RANGE, density=True, histtype='step', color='red', label='Resampled nonResonant MC')
    #plt.errorbar(bin_centers, np_hist, yerr=np.sqrt(np_hist), color='red', marker='')
    plt.plot(x_trial, y_fit, color='blue', label='Fit - Exponential + Gaussian@125GeV')
    plt.legend()
    plt.savefig(file[:-3]+'png')
    plt.close()
