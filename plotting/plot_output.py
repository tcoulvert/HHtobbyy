# Stdlib packages
import argparse
import copy
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
from plotting_utils import plot_filepath
from retrieval_utils import (
    get_class_sample_map, get_n_folds, get_train_DMatrices
)
from training_utils import (
    get_model_func, get_dataset_filepath
)
from evaluation_utils import (
    evaluate, transform_preds_func, transform_preds_options
)

################################


CWD = os.getcwd()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath",
    help="Full filepath on LPC for trained model files"
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
    "--discriminator", 
    choices=transform_preds_options(),
    default=transform_preds_options()[0],
    help="Defines the discriminator to use for output scores, discriminators are implemented in evaluation_utils"
)

################################


plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_output_scores(
    class_preds: dict, plot_name, plot_dirpath,
    plot_prefix='', plot_postfix='', bins=1000, log=False, weights=None
):
    plt.figure(figsize=(9,7))

    hist_axis = hist.axis.Regular(bins, 0., 1., name='var', growth=False, underflow=False, overflow=False)
    hists, labels = [], []
    for class_name, class_pred in class_preds.items():
        hists.append(
            hist.Hist(hist_axis, storage='weight').fill(
                var=class_pred, 
                weight=weights[sample_name] if weights is not None else np.ones_like(sigs_and_bkgs[sample_name])
            )
        )
        labels.append(class_name)
    hep.histplot(
        hists,
        yerr=(True if weights is not None else False),
        alpha=0.8, density=(False if weights is not None else True), histtype='step',
        label=labels
    )

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Output score')
    if log:
        plt.yscale('log')
    
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()

def make_output_scores(training_dirpath: str, dataset_dirpath: str, dataset: str, discriminator: str):
    plot_dirpath = os.path.join(training_dirpath, "plots", "output_scores")
    if not os.path.exists(plot_dirpath):
        os.makedirs(plot_dirpath)

    base_tpr = np.linspace(0, 1, 5000)

    get_booster = get_model_func(training_dirpath)
    CLASS_SAMPLE_MAP = get_class_sample_map(dataset_dirpath)
    CLASS_NAMES = [key for key in CLASS_SAMPLE_MAP.keys()]

    transform_labels, transform_preds = transform_preds_func(CLASS_NAMES, discriminator)

    plot_data = {
        class_name: {'preds': list(), 'labels': list(), 'weights': list()}
        for class_name in CLASS_NAMES
    }

    # plot ROCs
    for fold_idx in range(get_n_folds(dataset_dirpath)):
        booster = get_booster(fold_idx)

        fold_plot_data =  {
            class_name: {'preds': None, 'labels': None, 'weights': None}
            for class_name in CLASS_NAMES
        }

        for j, class_name in enumerate(CLASS_NAMES):
            train_dm, _, test_dm = get_train_DMatrices(dataset_dirpath, fold_idx)

            if dataset == "train-test": dm = test_dm
            elif dataset == "train": dm = train_dm

            event_mask = (dm.get_label() == j)

            nD_preds = evaluate(booster, dm)[event_mask]
            transformed_preds = transform_preds(nD_preds)

            # Add preds to full list for cross-fold evaluation
            fold_plot_data[class_name]['preds'] = transformed_preds
            fold_plot_data[class_name]['weights'] = dm.get_weight()[event_mask]
            for data_name, data in fold_plot_data[class_name].items():
                if data is None: continue
                plot_data[class_name][data_name].extend(data)

        for pred_idx, pred_label in enumerate(transform_labels):
            plot_output_scores({class_name: {'preds': class_data['preds'][:, pred_idx], } for class_name, class_data in fold_plot_dat.items()}, f"")

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

    make_output_scores(training_dirpath, dataset_dirpath, args.dataset, args.discriminator)