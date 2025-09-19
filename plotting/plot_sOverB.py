# %matplotlib widget
# Stdlib packages
import copy
import datetime
import glob
import json
import os
import re
import warnings
from pathlib import Path

# Common Py packages
import awkward as ak
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.special import logit as inverse_sigmoid

# HEP packages
import gpustat
import h5py
import hist
import mplhep as hep
import xgboost as xgb
from cycler import cycler

# ML packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
from sklearn.metrics import log_loss
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Module packages
from plotting_utils import (
    plot_filepath,
)

################################


gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_s_over_root_b(
    sig, bkg, label, plot_name, plot_dirpath,
    plot_prefix='', plot_postfix='', bins=1000, weights={'sig': None, 'bkg': None},
    lines=None, lines_labels=None, line_colors=None, arctanh=False
):
    plt.figure(figsize=(9,7))

    if arctanh:
        end_point = 6.
        hist_axis = hist.axis.Regular(bins, 0., end_point, name='var', growth=False, underflow=False, overflow=False)
    else:
        end_point = 1.
        hist_axis = hist.axis.Regular(bins, 0., end_point, name='var', growth=False, underflow=False, overflow=False)
    sig_hist = hist.Hist(hist_axis, storage='weight').fill(var=sig, weight=weights['sig'] if weights['sig'] is not None else np.ones_like(sig))
    bkg_hist = hist.Hist(hist_axis, storage='weight').fill(var=bkg, weight=weights['bkg'] if weights['bkg'] is not None else np.ones_like(bkg))
    s_over_root_b_points = sig_hist.values().flatten() / np.sqrt(bkg_hist.values().flatten())
    plt.plot(
        np.arange(0., end_point, end_point*(1/bins)), s_over_root_b_points, 
        label=f'{label} - s/√b', alpha=0.8
    )

    if lines is not None:
        for i in range(len(lines)):
            plt.vlines(
                lines[i], 0, np.max(s_over_root_b_points), 
                label='s/√b'+(' - '+lines_labels[i] if lines_labels is not None else ''), 
                alpha=0.5, colors=line_colors[i]
            )
    
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Output score')
    plt.ylabel('s/√b')
    
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()

def optimize_cut_boundaries(sigs, bkgs, weights, bins=10000, arctanh=False):
    hist_list_fold = []
    cut_boundaries_fold = []
    cut_s_over_root_bs_fold = []
    sig_weights_fold = []
    bkg_weights_fold = []
    if len(np.shape(sigs)) == 1:
        sigs, bkgs = [sigs], [bkgs] 
    if arctanh:
        end_point = 6.
    else:
        end_point = 1.
    for sig, bkg in zip(sigs, bkgs):
        hist_axis = hist.axis.Regular(bins, 0., end_point, name='var', growth=False, underflow=False, overflow=False)
        sig_hist = hist.Hist(hist_axis, storage='weight').fill(var=sig, weight=weights['sig'])
        bkg_hist = hist.Hist(hist_axis, storage='weight').fill(var=bkg, weight=weights['bkg'])
        hist_list_fold.append({'sig': copy.deepcopy(sig_hist), 'bkg': copy.deepcopy(bkg_hist)})

        fold_idx_cuts_bins_inclusive = []
        fold_idx_sig_weights = []
        fold_idx_bkg_weights = []
        fold_idx_prev_s_over_root_b = []
        prev_s_over_root_b = 0
        for i in range(bins):
            s = np.sum(sig_hist.values().flatten()[
                (bins-1) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
            ])
            sqrt_b = np.sqrt(np.sum(bkg_hist.values().flatten()[
                (bins-1) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
            ]))
            if prev_s_over_root_b < (s / sqrt_b) or s < 0.25:
                prev_s_over_root_b = s / sqrt_b
                continue
            else:
                fold_idx_sig_weights.append(
                    {
                        'value': np.sum(sig_hist.values().flatten()[
                            (bins) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                        ]),
                        'w2': np.sqrt(np.sum(sig_hist.variances().flatten()[
                            (bins) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                        ])),
                    }
                )
                fold_idx_bkg_weights.append(
                    {
                        'value': np.sum(bkg_hist.values().flatten()[
                            (bins) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                        ]),
                        'w2': np.sqrt(np.sum(bkg_hist.variances().flatten()[
                            (bins) - i : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                        ])),
                    }
                )
                fold_idx_cuts_bins_inclusive.append(bins - i)
                fold_idx_prev_s_over_root_b.append(prev_s_over_root_b)
                prev_s_over_root_b = 0
        fold_idx_sig_weights.append(
            {
                'value': np.sum(sig_hist.values().flatten()[
                    0 : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                ]),
                'w2': np.sqrt(np.sum(sig_hist.variances().flatten()[
                    0 : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                ])),
            }
        )
        fold_idx_bkg_weights.append(
            {
                'value': np.sum(bkg_hist.values().flatten()[
                    0 : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                ]),
                'w2': np.sqrt(np.sum(bkg_hist.variances().flatten()[
                    0 : bins if len(fold_idx_cuts_bins_inclusive) == 0 else fold_idx_cuts_bins_inclusive[-1]
                ])),
            }
        )
        fold_idx_cuts_bins_inclusive.append(0)
        fold_idx_prev_s_over_root_b.append(prev_s_over_root_b)
        fold_idx_score_cuts = [end_point * (bin_i / bins) for bin_i in fold_idx_cuts_bins_inclusive]
        cut_boundaries_fold.append(fold_idx_score_cuts)
        cut_s_over_root_bs_fold.append(fold_idx_prev_s_over_root_b)
        sig_weights_fold.append(fold_idx_sig_weights)
        bkg_weights_fold.append(fold_idx_bkg_weights)
    return cut_boundaries_fold, cut_s_over_root_bs_fold, sig_weights_fold, bkg_weights_fold

def p_to_xyz(p, split=True):  # makes a tetrahedron with height 1 and vertices {(0, 0, 0),  (√3/2, 0, √3/2),  (0, √3/2, √3/2),  (√3/2, √3/2, 0)}
    rt3o2 = np.sqrt(3) / 2

    x = rt3o2 * (0*p[:, 0] + p[:, 1] + p[:, 2] + 0*p[:, 3])
    y = rt3o2 * (0*p[:, 0] + 0*p[:, 1] + p[:, 2] + p[:, 3])
    z = rt3o2 * (0*p[:, 0] + p[:, 1] + 0*p[:, 2] + p[:, 3])

    if split:
        return x, y, z
    else:
        return np.column_stack((x, y, z))
    
def tetrahedron_lines(split=True):
    sig_to_ttH_like    = np.array([np.sqrt(3)/2,         0,            np.sqrt(3)/2])
    sig_to_VH_like     = np.array([0,                np.sqrt(3)/2,     np.sqrt(3)/2])
    sig_to_nonRes_like = np.array([np.sqrt(3)/2,     np.sqrt(3)/2,                0])

    if split:
        return sig_to_ttH_like, sig_to_VH_like, sig_to_nonRes_like
    else:
        return np.column_stack((sig_to_ttH_like, sig_to_VH_like, sig_to_nonRes_like))

def output_to_3d_thresholds(p, split=True):
    data = p_to_xyz(p, split=False)
    tetrahedron_matrix = tetrahedron_lines(split=False)

    thresholds = np.einsum('ij,jk', data, tetrahedron_matrix)

    if split:
        return thresholds[:, 0], thresholds[:, 1], thresholds[:, 2]
    else:
        return thresholds

def optimize_cuts(
    preds: np.ndarray, labels: np.ndarray, weights: np.ndarray,
    param_names=['r1', 'r2', 'r3'], param_range=[(0., 1.), (0., 1.), (0., 1.)], 
    n_steps=int(5e2), verbose: bool=False, min_sig: float=0.25, prefactor: float=1e3, rng_seed: int=21,
    fit_funcs_per_param: list=['power_law', 'power_law', 'power_law']
):
    # tetrahedron bkg vectors
    sig_to_ttH_like, sig_to_VH_like, sig_to_nonRes_like = tetrahedron_lines()
    tetrahedron_matrix = tetrahedron_lines(split=False)

    # 3D outputs
    xyz_preds = p_to_xyz(preds, split=False)
    xyz_thresholds = np.einsum('ij,jk', xyz_preds, tetrahedron_matrix)

    sig_to_ttH_preds = np.einsum('ij,j', xyz_preds, sig_to_ttH_like)
    sig_to_VH_preds = np.einsum('ij,j', xyz_preds, sig_to_VH_like)
    sig_to_nonRes_preds = np.einsum('ij,j', xyz_preds, sig_to_nonRes_like)

    # histogramed counts
    sig_to_ttH_counts, sig_to_ttH_bins = np.histogram(sig_to_ttH_preds[labels == 0], bins=1000, range=(0., 0.8), density=True)
    sig_to_VH_counts, sig_to_VH_bins = np.histogram(sig_to_VH_preds[labels == 0], bins=1000, range=(0., 0.8), density=True)
    sig_to_nonRes_counts, sig_to_nonRes_bins = np.histogram(sig_to_nonRes_preds[labels == 0], bins=1000, range=(0., 0.8), density=True)

    # shift to center of bins
    def bin_centers(bins_array):
        return np.array([np.mean([bins_array[bin_i], bins_array[bin_i+1]]) for bin_i in range(len(bins_array)-1)])
    
    sig_to_ttH_bin_centers = bin_centers(sig_to_ttH_bins)
    sig_to_VH_bin_centers = bin_centers(sig_to_VH_bins)
    sig_to_nonRes_bin_centers = bin_centers(sig_to_nonRes_bins)

    # fit data
    def get_fit_funcs_per_param():
        func_list, fit_func_list, transform_func_list = [], [], []
        for func_name in fit_funcs_per_param:
            if func_name == 'power_law':
                func_list.append(power_law)
                fit_func_list.append(fit_power_law)
                transform_func_list.append(power_law_transform)
            elif func_name == 'exponential':
                func_list.append(exponential)
                fit_func_list.append(fit_exponential)
                transform_func_list.append(exponential_transform)
            elif func_name == 'levy':
                func_list.append(levy)
                fit_func_list.append(fit_levy)
                transform_func_list.append(levy_transform)
            else:
                raise Exception(f"Fit function requested is not implemented. You asked for {func_name}, the implemented functions are power_law, exponential, and levy.")
        return func_list, fit_func_list, transform_func_list
    
    power_law = lambda x, a, k: a * np.power(x, -k)
    def fit_power_law(x, y, sigma=None):
        a_init = 1.
        k_init = 5.

        opt_params, opt_cov  = curve_fit(
            power_law, 
            x,
            y,
            p0=[a_init, k_init],
            sigma=sigma
        )
        return opt_params, opt_cov
    power_law_transform = lambda a, k: (
        lambda x: ((-k + 1) / a) * (x ** (1 / (-k + 1)))
    )

    exponential = lambda x, a, k: a * np.exp(-x * k)
    def fit_exponential(x, y, sigma=None):
        a_init = y[0]
        k_init = 1 / np.mean([
            x[y > (y[0] / np.exp(1))][-1],
            x[y < (y[0] / np.exp(1))][0],
        ])

        opt_params, opt_cov = curve_fit(
            exponential, 
            x,
            y,
            p0=[a_init, k_init],
            sigma=sigma
        )
        return opt_params, opt_cov
    exponential_transform = lambda a, k: (
        lambda x: (-1 / k) * np.log(-k * x / a)
    )

    levy = lambda x, c, mu: (
        np.sqrt(c / (2 * np.pi)) * np.exp(-c / (2 * (x - mu))) / np.power(x - mu, 3/2)
    )
    def fit_levy(x, y, sigma=None):
        c_init = 0.01  # success of fit withint 600 tries (kwarg of curve_fit) HIGHLY
        mu_init = 0.   #  sensitive to these initial choices of parameters. maybe rescale them by 1,000?

        opt_params, opt_cov = curve_fit(
            levy, 
            x,
            y,
            p0=[c_init, mu_init],
            sigma=sigma,
        )
        return opt_params, opt_cov
    gaussian_transfrom = lambda x: (10 / np.log(41)) * np.log(
        1 - (np.log(-np.log2(x)) / np.log(22))
    )  # approximation taken from https://dmi.units.it/~soranzo/epureAMS85-88-2014%202.pdf
    levy_transform = lambda c, mu: (
        lambda x: (c / np.power(gaussian_transfrom(1 - x/2), 2)) + mu
    )
    
    funcs, fit_funcs, transform_funcs = get_fit_funcs_per_param()
    sig_to_ttH_popt_func, sig_to_ttH_popt_cov = fit_funcs[0](sig_to_ttH_bin_centers, sig_to_ttH_counts, sigma=1/np.power(sig_to_ttH_counts, 1/2))
    sig_to_VH_popt_func, sig_to_VH_popt_cov = fit_funcs[1](sig_to_VH_bin_centers, sig_to_VH_counts, sigma=1/np.power(sig_to_VH_counts, 1/2))
    sig_to_nonRes_popt_func, sig_to_nonRes_popt_cov = fit_funcs[2](sig_to_nonRes_bin_centers, sig_to_nonRes_counts, sigma=1/np.power(sig_to_nonRes_counts, 1/2))
    sig_to_ttH_transform = transform_funcs[0](*sig_to_ttH_popt_func)
    sig_to_VH_transform = transform_funcs[1](*sig_to_VH_popt_func)
    sig_to_nonRes_transform = transform_funcs[2](*sig_to_nonRes_popt_func)

    sig_to_ttH_func = funcs[0](sig_to_ttH_bin_centers, *sig_to_ttH_popt_func)
    sig_to_VH_func = funcs[1](sig_to_ttH_bin_centers, *sig_to_VH_popt_func)
    sig_to_nonRes_func = funcs[2](sig_to_ttH_bin_centers, *sig_to_nonRes_popt_func)
    plt.figure()
    plt.plot(sig_to_ttH_bin_centers, sig_to_ttH_counts, alpha=0.7, color='r', linestyle='-', label='Sig to ttH-like')
    plt.plot(sig_to_VH_bin_centers, sig_to_VH_counts, alpha=0.7, color='b', linestyle='-', label='Sig to VH-like')
    plt.plot(sig_to_nonRes_bin_centers, sig_to_nonRes_counts, alpha=0.7, color='g', linestyle='-', label='Sig to nonRes-like')
    plt.legend()
    plt.show()
    
    if np.all(np.array(fit_funcs_per_param) == 'levy'):
        print(f"opt ttH levy values: c = {sig_to_ttH_popt_func[0]}, $\mu$ = {sig_to_ttH_popt_func[1]}")
        print(f"opt VH levy values: c = {sig_to_VH_popt_func[0]}, $\mu$ = {sig_to_VH_popt_func[1]}")
        print(f"opt nonRes levy values: c = {sig_to_nonRes_popt_func[0]}, $\mu$ = {sig_to_nonRes_popt_func[1]}")

    plt.figure()
    plt.plot(sig_to_ttH_bin_centers, sig_to_ttH_counts, alpha=0.7, color='r', linestyle='-', label='Sig to ttH-like')
    plt.plot(sig_to_VH_bin_centers, sig_to_VH_counts, alpha=0.7, color='b', linestyle='-', label='Sig to VH-like')
    plt.plot(sig_to_nonRes_bin_centers, sig_to_nonRes_counts, alpha=0.7, color='g', linestyle='-', label='Sig to nonRes-like')
    plt.plot(sig_to_ttH_bin_centers, sig_to_ttH_func, alpha=0.7, color='r', linestyle='--', label='Sig to ttH-like - Fit exp')
    plt.plot(sig_to_VH_bin_centers, sig_to_VH_func, alpha=0.7, color='b', linestyle='--', label='Sig to VH-like - Fit exp')
    plt.plot(sig_to_nonRes_bin_centers, sig_to_nonRes_func, alpha=0.7, color='g', linestyle='--', label='Sig to nonRes-like - Fit exp')
    plt.legend()
    plt.show()

    space  = [Real(float(param_range[i][0]), float(param_range[i][1]), "uniform", name=param_name) for i, param_name in enumerate(param_names)]
    def space_transform(X):
        return [
            sig_to_ttH_transform(X[param_names[0]]),
            sig_to_VH_transform(X[param_names[1]]),
            sig_to_nonRes_transform(X[param_names[2]]),
        ]

    @use_named_args(space)
    def objective(**X):
        thresholds = space_transform(X)
        if verbose:
            print("New configuration: {}".format(thresholds))
        sample_mask = np.all(xyz_thresholds < thresholds, axis=1)

        num_sig = np.abs(
            np.sum(
                weights[
                    np.logical_and(
                        labels == 0,
                        sample_mask
                    )
                ]
            )
        )
        num_singleH_bkg = np.abs(
            np.sum(
                weights[
                    np.logical_and(
                        np.logical_or(
                            labels == 1,
                            labels == 2
                        ),
                        sample_mask
                    )
                ]
            )
        )
        num_nonRes_bkg = np.abs(
            np.sum(
                weights[
                    np.logical_and(
                        labels == 3,
                        sample_mask
                    )
                ]
            )
        )
        singleH_to_nonRes_factor = 3.
        num_rescale_bkg = (
            singleH_to_nonRes_factor * num_singleH_bkg
        ) + num_nonRes_bkg
        num_bkg = num_singleH_bkg + num_nonRes_bkg

        def s_over_b(s, b, case='realistic'):
            if case == 'simplistic':
                return s / np.sqrt(b)
            elif case == 'realistic':
                return np.sqrt(
                    2 * (
                        (s + b) * np.log(1 + (s / b)) - s
                    )
                )
        s_over_root_b = s_over_b(num_sig, num_bkg)
        opt_criteria = s_over_b(num_sig, num_rescale_bkg)

        if num_sig == 0 and num_bkg == 0:
            both_0 = prefactor*1e1
            if verbose:
                print(f"both sig and bkg 0 at this hyperplane => {both_0}")
                print('='*60)
            return both_0
        elif num_sig < min_sig:
            small_sig = prefactor*0
            if verbose:
                print(f"too little sig ({num_sig}) at this hyperplane => {small_sig}")
                print('='*60)
            return small_sig
        elif num_bkg == 0:
            zero_bkg = -prefactor*num_sig
            if verbose:
                print(f"zero bkg at this hyperplane (likely due to finite data rather than real bkg-free zone) => {zero_bkg}")
                print('='*60)
            return zero_bkg
        
        if verbose:
            print(f"s = {num_sig}, b = {num_bkg}, s/√b = {s_over_root_b} => {-prefactor*opt_criteria}")
            print('='*60)

        return -prefactor*opt_criteria
    
    res_gp = gp_minimize(
        objective, space, random_state=rng_seed, 
        n_calls=n_steps, n_initial_points=n_steps-100,
        n_restarts_optimizer=5
    )

    opt_params = [float(res_gp.x[i]) for i in range(len(space))]
    opt_cuts = [float(opt_cut) for opt_cut in space_transform({param_names[i]: res_gp.x[i] for i in range(len(param_names))})]
    if verbose:
        print("Best parameters: {}".format(opt_cuts))
        print(f"Best s/√b = {-res_gp.fun / prefactor}")

    return opt_cuts, opt_params


def multi_optimize_cut_boundaries(preds: list, labels: np.ndarray, weights: np.ndarray, num_categories: int=3, min_sig: float=0.25, n_steps: int=200):
    clf_dict = {}
    param_clf_dict = {}
    # clf_dict[0] = [0.008485205139697272, 0.03976034437493922, 0.06505303760998234]
    # for cat in range(1, num_categories):
    for cat in range(num_categories):

        clf_dict[cat] = []
        param_clf_dict[cat] = []

        if cat == 0:
            opt_cuts, opt_params = optimize_cuts(
                np.array(preds), labels, weights, verbose=True,
                n_steps=n_steps, min_sig=min_sig, rng_seed=None,
            )

        else:
            slice_array = np.ones_like(labels, dtype=bool)
            for prev_cat in range(cat):
                slice_array = np.logical_and(
                    slice_array,
                    np.logical_not(
                        np.all(
                            output_to_3d_thresholds(np.array(preds), split=False) < clf_dict[prev_cat], 
                            axis=1
                        )
                    )
                )

            sliced_preds = np.array(preds)[slice_array]
            sliced_labels = labels[slice_array]
            sliced_weights = weights[slice_array]
            
            opt_cuts, opt_params = optimize_cuts(
                sliced_preds, sliced_labels, sliced_weights, verbose=True,
                n_steps=n_steps, min_sig=min_sig, rng_seed=None,
                fit_funcs_per_param=['levy', 'levy', 'levy']
            )

        clf_dict[cat] = opt_cuts
        param_clf_dict[cat] = opt_params

    return clf_dict