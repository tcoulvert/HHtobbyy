# %matplotlib widget
# Stdlib packages
import copy
import datetime
import glob
import json
import os
import re
import subprocess
import sys
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

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
sys.path.append(os.path.join(GIT_REPO, "training/"))

# Module packages
from plotting_utils import (
    plot_filepath, 
    get_ttH_score, get_QCD_score,
    cat_mask
)
from retrieval_utils import (
    get_DMatrices
)
from training_utils import (
    get_filepaths_func
)

################################


gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

order = ['ggF HH', 'ttH + bbH', 'VH', 'non-res + ggFH + VBFH']

################################


def plot_rocs(
    fprs, tprs, labels, plot_name, plot_dirpath,
    plot_prefix='', plot_postfix='', close=True, log=None
):
    plt.figure(figsize=(9,7))
    
    for fpr, tpr, label in zip(fprs, tprs, labels):
        plt.plot(fpr, tpr, label=label, linestyle='solid')

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Background contamination')
    plt.ylabel('Signal efficiency')
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    if log is not None and re.search('x', log) is not None:
        plt.xscale('log')
        plt.xlim((1e-5, 1))
    if log is not None and re.search('y', log) is not None:
        plt.yscale('log')
        plt.ylim((1e-4, 1))
    
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    if close:
        plt.close()

def make_rocs(output_dirpath: str, base_filepath: str):
    output_dirpath = os.path.join(output_dirpath, "")

    plot_dirpath = os.path.join(output_dirpath, "plots", "2D_ROCs")
    if not os.path.exists(plot_dirpath):
        os.makedirs(plot_dirpath)

    base_tpr = np.linspace(0, 1, 5000)
    preds, truths, weights = {sample_name: list() for sample_name in order}, {sample_name: list() for sample_name in order}, {sample_name: list() for sample_name in order}

    param_filepath = os.path.join(output_dirpath, f"{output_dirpath.split('/')[-2]}_best_params.json")
    with open(param_filepath, 'r') as f:
        param = json.load(f)
    param = list(param.items()) + [('eval_metric', 'mlogloss')]

    get_filepaths_lambda_filepath = os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_get_filepaths_lambda.txt")
    with open(get_filepaths_lambda_filepath, 'r') as f:
        get_filepaths_lambda_str = f.read()
    if base_filepath is None:
        base_filepath = 

    # plot ROCs
    for fold_idx in range(5):
        booster = xgb.Booster(param)
        booster.load_model(os.path.join(output_dirpath, f"{output_dirpath.split('/')[-2]}_BDT_fold{fold_idx}.model"))

        fold_fprs_tth, fold_tprs_tth, fold_thresholds_tth, fold_auc_tth = [], [], [], []
        fold_fprs_qcd, fold_tprs_qcd, fold_thresholds_qcd, fold_auc_qcd = [], [], [], []

        _, _, test_dm = get_DMatrices(
            get_filepaths_func(base_filepath), fold_idx
        )

        for j, sample_name in enumerate(order):

            if j == 0:
                event_mask = (test_dm.get_label() > -1)
            else:
                event_mask = np.logical_or(
                    test_dm.get_label() == 0,
                    test_dm.get_label() == j
                )

            full_preds = booster.predict(
                test_dm, iteration_range=(0, booster.best_iteration+1)
            )[event_mask]
            preds[sample_name].extend(full_preds.tolist())

            tth_preds = get_ttH_score(full_preds)
            qcd_preds = get_QCD_score(full_preds)

            signal_truths = (test_dm.get_label() == 0)[event_mask]
            truths[sample_name].extend(test_dm.get_label()[event_mask].tolist())

            # weights[sample_name].extend(weights_plot_test[f"fold_{fold_idx}"][event_mask].tolist())

            # fpr_tth, tpr_tth, threshold_tth = roc_curve(signal_truths, tth_preds, sample_weight=weights_plot_test[f"fold_{fold_idx}"][event_mask])
            # fpr_qcd, tpr_qcd, threshold_qcd = roc_curve(signal_truths, qcd_preds, sample_weight=weights_plot_test[f"fold_{fold_idx}"][event_mask])

            fpr_tth, tpr_tth, threshold_tth = roc_curve(signal_truths, tth_preds)
            fpr_qcd, tpr_qcd, threshold_qcd = roc_curve(signal_truths, qcd_preds)

            fpr_tth = np.interp(base_tpr, tpr_tth, fpr_tth)
            threshold_tth = np.interp(base_tpr, tpr_tth, threshold_tth)
            auc_tth = float(trapezoid(base_tpr, fpr_tth))

            fpr_qcd = np.interp(base_tpr, tpr_qcd, fpr_qcd)
            threshold_qcd = np.interp(base_tpr, tpr_qcd, threshold_qcd)
            auc_qcd = float(trapezoid(base_tpr, fpr_qcd))

            fold_fprs_tth.append(fpr_tth)
            fold_tprs_tth.append(base_tpr)
            fold_thresholds_tth.append(threshold_tth)
            fold_auc_tth.append(auc_tth)

            fold_fprs_qcd.append(fpr_qcd)
            fold_tprs_qcd.append(base_tpr)
            fold_thresholds_qcd.append(threshold_qcd)
            fold_auc_qcd.append(auc_qcd)

        labels_tth = [
            f"{sample_name} DttH, AUC = {fold_auc_tth[i]:.4f}" 
            for i, sample_name in enumerate(order)
        ]
        labels_qcd = [
            f"{sample_name} DQCD, AUC = {fold_auc_qcd[i]:.4f}" 
            for i, sample_name in enumerate(order)
        ]

        plot_rocs(fold_fprs_tth, fold_tprs_tth, labels_tth, f"DttH_logroc_weighted_fold{fold_idx}", plot_dirpath, log='x')
        plot_rocs(fold_fprs_qcd, fold_tprs_qcd, labels_qcd, f"DQCD_logroc_weighted_fold{fold_idx}", plot_dirpath, log='x')

    fprs_tth, tprs_tth, thresholds_tth, aucs_tth = [], [], [], []
    fprs_qcd, tprs_qcd, thresholds_qcd, aucs_qcd = [], [], [], []
    for j, sample_name in enumerate(order):

        tth_preds = get_ttH_score(np.array(preds[sample_name]))
        qcd_preds = get_QCD_score(np.array(preds[sample_name]))

        signal_truths = (np.array(truths[sample_name]) == 0)

        fpr_tth, tpr_tth, threshold_tth = roc_curve(signal_truths, tth_preds)
        fpr_qcd, tpr_qcd, threshold_qcd = roc_curve(signal_truths, qcd_preds)

        def sqrtNerr(num, denom):
            return ( (num / denom**2) + (num**2 / denom**3) )**0.5  # Poisson error

        fptidx = np.argmin(np.abs(fpr_tth - 1e-2))
        fptstat = sqrtNerr(np.sum(np.logical_and(signal_truths > 0, tth_preds > threshold_tth[fptidx])), np.sum(signal_truths > 0))
        fpqidx = np.argmin(np.abs(fpr_qcd - 5e-5))
        fpqstat = sqrtNerr(np.sum(np.logical_and(signal_truths > 0, qcd_preds > threshold_tth[fpqidx])), np.sum(signal_truths > 0))
        fpqlooseidx = np.argmin(np.abs(fpr_qcd - 1e-3))
        fpqloosestat = sqrtNerr(np.sum(np.logical_and(signal_truths > 0, qcd_preds > threshold_tth[fpqlooseidx])), np.sum(signal_truths > 0))

        head_str = ' - '.join(output_dirpath.split('/')[-3:-1])+'\n' if sample_name == order[0] else ''
        print_str = (
            head_str
            + f"sig vs. {sample_name if sample_name != order[0] else 'all'}"
            + '\n' + f"  * DttH - signal tpr = {tpr_tth[fptidx]:.3f}±{fptstat:.3f} @ fpr of {fpr_tth[fptidx]:.3f}"
            + '\n' + f"  * DQCD - signal tpr = {tpr_qcd[fpqlooseidx]:.4f}±{fpqloosestat:.4f} @ fpr of {fpr_qcd[fpqlooseidx]:.4f}"
            + '\n' + f"  * DQCD - signal tpr = {tpr_qcd[fpqidx]:.5f}±{fpqstat:.5f} @ fpr of {fpr_qcd[fpqidx]:.5f}"
        )
        print(print_str)

        fpr_tth = np.interp(base_tpr, tpr_tth, fpr_tth)
        threshold_tth = np.interp(base_tpr, tpr_tth, threshold_tth)
        auc_tth = float(trapezoid(base_tpr, fpr_tth))

        fpr_qcd = np.interp(base_tpr, tpr_qcd, fpr_qcd)
        threshold_qcd = np.interp(base_tpr, tpr_qcd, threshold_qcd)
        auc_qcd = float(trapezoid(base_tpr, fpr_qcd))

        fprs_tth.append(fpr_tth)
        tprs_tth.append(base_tpr)
        thresholds_tth.append(threshold_tth)
        aucs_tth.append(auc_tth)

        fprs_qcd.append(fpr_qcd)
        tprs_qcd.append(base_tpr)
        thresholds_qcd.append(threshold_qcd)
        aucs_qcd.append(auc_qcd)

    labels_tth = [
        f"{sample_name} DttH, AUC = {fold_auc_tth[i]:.4f}" 
        for i, sample_name in enumerate(order)
    ]
    labels_qcd = [
        f"{sample_name} DQCD, AUC = {fold_auc_qcd[i]:.4f}" 
        for i, sample_name in enumerate(order)
    ]

    plot_rocs(fprs_tth, tprs_tth, labels_tth, f"DttH_logroc_weighted_sum", plot_dirpath, log='xy')
    plot_rocs(fprs_qcd, tprs_qcd, labels_qcd, f"DQCD_logroc_weighted_sum", plot_dirpath, log='xy')

if __name__ == "__main__":
    make_rocs(sys.argv[1], sys.argv[2])