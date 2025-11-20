# Stdlib packages
import argparse
import json
import logging
import os
import re
import subprocess
import sys

# Common Py packages
import numpy as np
from matplotlib import pyplot as plt

# HEP packages
import mplhep as hep
import xgboost as xgb
from cycler import cycler

# ML packages
from sklearn.metrics import auc, roc_curve
from scipy.integrate import trapezoid

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
    plot_filepath, 
    get_ttH_score, get_QCD_score,
)
from retrieval_utils import (
    get_class_sample_map, get_train_DMatrices
)
from training_utils import (
    get_model_func, get_dataset_filepath
)
from evaluation_utils import (
    get_filepaths, evaluate
)

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "--training_dirpath", 
    default=CWD,
    help="Full filepath on LPC for trained model files"
)
parser.add_argument(
    "--dataset_dirpath", 
    default=None,
    help="Full filepath on LPC for standardized dataset (train and test parquets), default is to use dataset in the training `dataset_dirpath.txt` file"
)
parser.add_argument(
    "--dataset", 
    choices=["train", "test", "train-test", "all"], 
    default="train-test",
    help="Evaluate and save out evaluation for what dataset"
)
parser.add_argument(
    "--syst_name", 
    choices=["nominal", "all"], 
    default="nominal",
    help="Evaluate and save out evaluation for what systematic of a dataset"
)

################################


plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

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

def make_rocs(training_dirpath: str, dataset_dirpath: str, dataset: str="train-test", syst_name: str="nominal"):
    plot_dirpath = os.path.join(training_dirpath, "plots", "2D_ROCs")
    if not os.path.exists(plot_dirpath):
        os.makedirs(plot_dirpath)

    base_tpr = np.linspace(0, 1, 5000)
    preds, truths = {}, {}

    get_booster = get_model_func(training_dirpath)
    CLASS_SAMPLE_MAP = get_class_sample_map(dataset_dirpath)
    CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

    # plot ROCs
    for fold_idx in range(5):
        booster = get_booster(fold_idx)

        preds, truths = {class_name: list() for class_name in CLASS_NAMES}, {class_name: list() for class_name in CLASS_NAMES}

        fold_fprs_tth, fold_tprs_tth, fold_thresholds_tth, fold_auc_tth = [], [], [], []
        fold_fprs_qcd, fold_tprs_qcd, fold_thresholds_qcd, fold_auc_qcd = [], [], [], []

        
        for j, class_name in enumerate(CLASS_NAMES):
            train_dm, val_dm, test_dm = get_train_DMatrices(dataset_dirpath, fold_idx)

            if j == 0:
                event_mask = (test_dm.get_label() > -1)
            else:
                event_mask = np.logical_or(
                    test_dm.get_label() == 0,
                    test_dm.get_label() == j
                )

            ROC_preds = evaluate(booster, test_dm)[event_mask]
            tth_preds = get_ttH_score(ROC_preds)
            qcd_preds = get_QCD_score(ROC_preds)

            signal_truths = (test_dm.get_label() == 0)[event_mask]

            # ttH ROC curve
            fpr_tth, tpr_tth, threshold_tth = roc_curve(signal_truths, tth_preds)
            fpr_tth = np.interp(base_tpr, tpr_tth, fpr_tth)
            threshold_tth = np.interp(base_tpr, tpr_tth, threshold_tth)
            auc_tth = float(trapezoid(base_tpr, fpr_tth))
            fold_fprs_tth.append(fpr_tth)
            fold_tprs_tth.append(base_tpr)
            fold_thresholds_tth.append(threshold_tth)
            fold_auc_tth.append(auc_tth)

            # QCD ROC curve
            fpr_qcd, tpr_qcd, threshold_qcd = roc_curve(signal_truths, qcd_preds)
            fpr_qcd = np.interp(base_tpr, tpr_qcd, fpr_qcd)
            threshold_qcd = np.interp(base_tpr, tpr_qcd, threshold_qcd)
            auc_qcd = float(trapezoid(base_tpr, fpr_qcd))
            fold_fprs_qcd.append(fpr_qcd)
            fold_tprs_qcd.append(base_tpr)
            fold_thresholds_qcd.append(threshold_qcd)
            fold_auc_qcd.append(auc_qcd)

            # Add preds to full list for cross-fold evaluation
            preds[class_name].extend(ROC_preds.tolist())
            truths[class_name].extend(test_dm.get_label()[event_mask].tolist())

        labels_tth = [
            f"{class_name} DttH, AUC = {fold_auc_tth[i]:.4f}" 
            for i, class_name in enumerate(CLASS_NAMES)
        ]
        labels_qcd = [
            f"{class_name} DQCD, AUC = {fold_auc_qcd[i]:.4f}" 
            for i, class_name in enumerate(CLASS_NAMES)
        ]

        plot_rocs(fold_fprs_tth, fold_tprs_tth, labels_tth, f"DttH_logroc_weighted_fold{fold_idx}", plot_dirpath, log='x')
        plot_rocs(fold_fprs_qcd, fold_tprs_qcd, labels_qcd, f"DQCD_logroc_weighted_fold{fold_idx}", plot_dirpath, log='x')

    fprs_tth, tprs_tth, thresholds_tth, aucs_tth = [], [], [], []
    fprs_qcd, tprs_qcd, thresholds_qcd, aucs_qcd = [], [], [], []
    for j, class_name in enumerate(CLASS_NAMES):

        tth_preds = get_ttH_score(np.array(preds[class_name]))
        qcd_preds = get_QCD_score(np.array(preds[class_name]))

        signal_truths = (np.array(truths[class_name]) == 0)

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

        head_str = ' - '.join(training_dirpath.split('/')[-3:-1])+'\n' if class_name == CLASS_NAMES[0] else ''
        print_str = (
            head_str
            + f"sig vs. {class_name if class_name != CLASS_NAMES[0] else 'all'}"
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
        f"{class_name} DttH, AUC = {fold_auc_tth[i]:.4f}" 
        for i, class_name in enumerate(CLASS_NAMES)
    ]
    labels_qcd = [
        f"{class_name} DQCD, AUC = {fold_auc_qcd[i]:.4f}" 
        for i, class_name in enumerate(CLASS_NAMES)
    ]

    plot_rocs(fprs_tth, tprs_tth, labels_tth, f"DttH_logroc_weighted_sum", plot_dirpath, log='xy')
    plot_rocs(fprs_qcd, tprs_qcd, labels_qcd, f"DQCD_logroc_weighted_sum", plot_dirpath, log='xy')

if __name__ == "__main__":
    args = parser.parse_args()

    training_dirpath = os.path.join(args.training_dirpath, "")
    if args.dataset_dirpath is None:
        dataset_dirpath = get_dataset_filepath(args.training_dirpath)
    else:
        dataset_dirpath = args.dataset_dirpath

    make_rocs(training_dirpath, dataset_dirpath, args.dataset, args.syst_name)