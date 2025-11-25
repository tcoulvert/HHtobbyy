# Stdlib packages
import argparse
import logging

# Common Py packages
import matplotlib.pyplot as plt

# HEP packages
import numpy as np
import pandas as pd
import scipy as scp

################################


from plotting_utils import make_plot_dirpath

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath",
    help="Full filepath on LPC for trained model files"
)

################################


N_BINS = 80
RANGE = (100, 180)


def resample_from_var(
    sample_var, sample_weight, n_events, 
    min_value=None,
    n_samples_per_event=1, bins=100, seed=None
):
    resample_rng = np.random.default_rng(seed=seed)

    np_hist, bin_edges = np.histogram(sample_var, bins=bins, weights=sample_weight, density=True)
    np_hist /= np.sum(np_hist)

    bin_choices = resample_rng.choice(np.arange(len(np_hist)), size=n_events*n_samples_per_event, p=np_hist)

    value_choices = (bin_edges[bin_choices+1] - bin_edges[bin_choices]) * resample_rng.random(size=n_events*n_samples_per_event) + bin_edges[bin_choices]

    if min_value is None or np.all(value_choices > min_value):
        return value_choices
    else:  # this is not really correct, just an approximation to make the code work faster
        bad_choices_bool = value_choices <= min_value

        largest_min_value = np.max(min_value[bad_choices_bool])

        np_hist, bin_edges = np.histogram(sample_var[sample_var > largest_min_value], bins=bins, weights=sample_weight[sample_var > largest_min_value], density=True)
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

def resample_grow_pd(var, bool_arr, n_duplicates_per_event):
    new_rows = pd.DataFrame(
        np.tile(
            ( var.loc[bool_arr] ).to_numpy(),
            (n_duplicates_per_event, 1)
        ),
        columns=var.columns
    )
    return pd.concat([var, new_rows], ignore_index=True)
        






resample_ = 10
RESAMPLE = 100  # Set to False for no resampling, otherwise sets the number of times to duplicate gjet data for resampling

plot_dirpath = os.path.join(OUTPUT_DIRPATH, "plots", "mass_sculpting_resample_single")
if not os.path.exists(plot_dirpath):
    os.makedirs(plot_dirpath)

score_cuts = [0.0, 0.7, 0.99, 0.9955]
label_arr = [f'score above {score_cut}' for score_cut in score_cuts]
plot_vars = ['mass', 'dijet_mass', 'HHbbggCandidate_mass']


BDT_perf_resample = [
    {
        f'preds{score_cut}': copy.deepcopy({plot_var: list() for plot_var in plot_vars+['event']}) for score_cut in score_cuts
    } for fold_idx in range(len(bdt_train_dict))
]

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
