# Stdlib packages
import argparse
import copy
import logging
import os
import subprocess
import sys

# Common Py packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# HEP packages
import mplhep as hep
from cycler import cycler

# ML packages
from sklearn.metrics import roc_curve
from scipy.integrate import trapezoid

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "training/"))
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))
sys.path.append(os.path.join(GIT_REPO, "evaluation/"))

# Module packages
from plotting_utils import (
    plot_filepath, make_plot_data, combine_prepostfix
)
from training_utils import get_dataset_dirpath
from retrieval_utils import get_class_sample_map
from evaluation_utils import (
    transform_preds_options, transform_preds_bkgeffs
)

################################


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
    choices=["train", "train-test", "data"], 
    default="train-test",
    help="Make output score distributions for what dataset"
)
parser.add_argument(
    "--weights", 
    action="store_true",
    help="Boolean to make plots using MC weights"
)
parser.add_argument(
    "--logx", 
    action="store_true",
    help="Boolean to make plots log-scale on x-axis"
)
parser.add_argument(
    "--logy", 
    action="store_true",
    help="Boolean to make plots log-scale on y-axis"
)
parser.add_argument(
    "--bins", 
    type=int,
    default=1000,
    help="Number of bins to use for histogram"
)
parser.add_argument(
    "--ROCtype", 
    choices=["OnevsRest", "OnevsOne"],
    default="OnevsRest",
    help="Defines how to compute the ROCs, \'OnevsRest\' means ROC curves of each class against all others, \'OnevsOne\' means ROC curves of each class against each other individually (only combinations, not permutations)"
)
parser.add_argument(
    "--discriminator", 
    choices=transform_preds_options(),
    default=transform_preds_options()[0],
    help="Defines the discriminator to use for output scores, discriminators are implemented in evaluation_utils"
)
parser.add_argument(
    "--save_FPRTPR", 
    action="store_true",
    help="Boolean to save out pandas file with TPR and FPR points inside"
)

################################


CWD = os.getcwd()
args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
if args.dataset_dirpath is None:
    DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
else:
    DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')
WEIGHTS = args.weights
LOGX = args.logx
LOGY = args.logy
BINS = args.bins
ROCTYPE = args.ROCtype
DISCRIMINATOR = args.discriminator
SAVE_FPRTPR = args.save_FPRTPR
BKGEFF = transform_preds_bkgeffs([key for key in get_class_sample_map(DATASET_DIRPATH).keys()], DISCRIMINATOR)
PLOT_TYPE = f'ROC_{ROCTYPE}_{DISCRIMINATOR}' + ('_unweighted' if not WEIGHTS else '_MCweighted') + ('_'+args.dataset.upper() if args.dataset != 'train-test' else '')

BASE_TPR = np.linspace(0, 1, 5000)

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_rocs(
    fprs, tprs, labels, plot_dirpath: str,
    plot_prefix: str='', plot_postfix: str=''
):
    if WEIGHTS:
        plot_postfix = combine_prepostfix(plot_postfix, 'MCweight', fixtype='postfix')
    if LOGX and LOGY:
        plot_postfix = combine_prepostfix(plot_postfix, 'logXY', fixtype='postfix')
    if LOGX: 
        plot_postfix = combine_prepostfix(plot_postfix, 'logX', fixtype='postfix')
    if LOGY: 
        plot_postfix = combine_prepostfix(plot_postfix, 'logY', fixtype='postfix')

    plt.figure(figsize=(9,7))

    FPRTPR_df = None
    for fpr, tpr, label in zip(fprs, tprs, labels):
        plt.plot(fpr, tpr, label=label, linestyle='solid')
        if SAVE_FPRTPR:
            new_df = pd.Dataframe([fpr, tpr], columns=['FPR: '+label, 'TPR: '+label])
            if FPRTPR_df is None: FPRTPR_df = copy.deepcopy(new_df)
            else: FPRTPR_df = pd.concat([FPRTPR_df, new_df])

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Background efficiency ($\epsilon_{{bkg}}$)')
    plt.ylabel('Signal efficiency ($\epsilon_{{sig}}$)')
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    if LOGX:
        plt.xscale('log')
        plt.xlim((1e-5, 1))
    if LOGY:
        plt.yscale('log')
        plt.ylim((1e-4, 1))
    
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    if SAVE_FPRTPR: FPRTPR_df.to_parquet(plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix, format='parquet'))
    plt.close()


def sigeff_at_bkgeff(sig, fpr, threshold, pred_idx):
    idx = np.argmin(np.abs(fpr - BKGEFF[pred_idx]))

    num = sig['weights'][(sig['preds'] > threshold[idx])]
    denom = sig['weights']

    sig_eff = np.sum(num) / np.sum(denom)
    sig_eff_err = np.sqrt( (np.sum(num**2) / np.sum(denom)**2) + (np.sum(num)**2 * np.sum(denom**2) / np.sum(denom)**4) )
    return sig_eff, sig_eff_err


def ROC_OnevsRest(
    plot_data: dict, plot_dirpath: str, pred_idx: int,
    plot_prefix: str='', plot_postfix: str=''
):
    fprs, tprs, plot_labels = list(), list(), list()
    roc_data = None
    for j, (class_name, class_data) in enumerate(plot_data.items()):
        if j == 0:
            roc_data = {
                data_name: np.concatenate([_class_data_[data_name] for _class_data_ in plot_data.values()])
                for data_name in class_data.keys()
            }

        fpr, tpr, threshold = roc_curve(
            roc_data['labels'], roc_data['preds'], pos_label=j, sample_weight=roc_data['weights'] if WEIGHTS else None
        )
        fprs.append(np.interp(BASE_TPR, tpr, fpr))
        tprs.append(BASE_TPR)
        
        epsilon_sig, sigma_epsilon_sig = sigeff_at_bkgeff({data_name: data[roc_data['labels'] == j] for data_name, data in roc_data.items()}, fpr, threshold, pred_idx)
        plot_labels.append(f"{class_name} vs. Rest - AUC = {float(trapezoid(tprs[-1], fprs[-1])):.3f}; $\epsilon_{{{class_name}}}$ = {epsilon_sig:.3f}±{sigma_epsilon_sig:.3f} @ $\epsilon_{{Rest}}$ = {BKGEFF[pred_idx]:e}")

    plot_rocs(fprs, tprs, plot_labels, plot_dirpath, plot_prefix=plot_prefix, plot_postfix=plot_postfix)

def ROC_OnevsOne(
    plot_data: dict, plot_dirpath: str, pred_idx: int,
    plot_prefix: str='', plot_postfix: str=''
):
    fprs, tprs, plot_labels = list(), list(), list()
    for j, (class_name, class_data) in enumerate(plot_data.items()):
        for k, (_class_name_, _class_data_) in enumerate(plot_data.items()):
            if k <= j: continue

            roc_data = {
                data_name: np.concatenate([class_data[data_name], _class_data_[data_name]])
                for data_name in class_data.keys()
            }

            fpr, tpr, threshold = roc_curve(
                roc_data['labels'], roc_data['preds'], pos_label=j, sample_weight=roc_data['weights'] if WEIGHTS else None
            )
            fprs.append(np.interp(BASE_TPR, tpr, fpr))
            tprs.append(BASE_TPR)

            epsilon_sig, sigma_epsilon_sig = sigeff_at_bkgeff({data_name: data[roc_data['labels'] == j] for data_name, data in roc_data.items()}, fpr, threshold, pred_idx)
            plot_labels.append(f"{class_name} vs. {_class_name_} - AUC = {float(trapezoid(tprs[-1], fprs[-1])):.3f}; $\epsilon_{{{class_name}}}$ = {epsilon_sig:.3f}±{sigma_epsilon_sig:.3f} @ $\epsilon_{{{_class_name_}}}$ = {BKGEFF[pred_idx]:e}")

    plot_rocs(fprs, tprs, plot_labels, plot_dirpath, plot_prefix=plot_prefix, plot_postfix=plot_postfix)


if __name__ == "__main__":
    if ROCTYPE == "OnevsRest":
        make_plot_data(TRAINING_DIRPATH, DATASET_DIRPATH, args.dataset, args.discriminator, PLOT_TYPE, ROC_OnevsRest, project_1D=True)
    elif ROCTYPE == "OnevsOne":
        make_plot_data(TRAINING_DIRPATH, DATASET_DIRPATH, args.dataset, args.discriminator, PLOT_TYPE, ROC_OnevsOne, project_1D=True)