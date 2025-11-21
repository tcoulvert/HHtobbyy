# Stdlib packages
import argparse
import json
import logging
import os
import subprocess
import sys

# Common Py packages
import numpy as np
from matplotlib import pyplot as plt

# HEP packages
import mplhep as hep
from cycler import cycler

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "training/"))
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))

# Module packages
from plotting_utils import (
    plot_filepath, pad_list, make_plot_dirpath, 
    combine_prepostfix
)
from training_utils import (
    get_dataset_dirpath, get_model_func
)
from retrieval_utils import get_n_folds

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Standardize BDT inputs and save out dataframe parquets.")
parser.add_argument(
    "training_dirpath",
    help="Full filepath on LPC for trained model files"
)

################################


CWD = os.getcwd()
args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
DATASET_DIRPATH = get_dataset_dirpath(TRAINING_DIRPATH)
PLOT_TYPE = 'loss'

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_train_val_losses(
    losses_arrs, labels, plot_name, plot_dirpath, 
    plot_prefix='', plot_postfix='', linestyles=None,
    losses_std_arrs=None
):
    plt.figure(figsize=(9,7))
    
    if type(losses_arrs[0]) is float:
        losses_arrs = [losses_arrs]
    if linestyles is None:
        linestyles = ['solid'] * len(losses_arrs)
    if labels is None:
        labels = [i for i in range(len(losses_arrs))]

    if losses_std_arrs is not None:
        for i in range(len(losses_std_arrs)):
            plt.fill_between(
                range(len(losses_std_arrs[i])), 
                losses_arrs[i]+losses_std_arrs[i], losses_arrs[i]-losses_std_arrs[i],
                alpha=0.7
            )

    for i in range(len(losses_arrs)):
        plt.plot(
            range(len(losses_arrs[i])), 
            losses_arrs[i], 
            label=f"{labels[i]} losses", linestyle=linestyles[i],
            alpha=0.7
        )
        
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('EPOCH')
    plt.ylabel('Data Loss')
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()


def make_losses():
    plot_dirpath = make_plot_dirpath(TRAINING_DIRPATH, PLOT_TYPE)

    with open(os.path.join(TRAINING_DIRPATH, f"{TRAINING_DIRPATH.split('/')[-2]}_BDT_eval_result.json"), 'r') as f:
        evals_result_dict = json.load(f)

    # plot train/val/test losses
    all_train, all_val, all_test = [], [], []
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        all_train.append(evals_result_dict[f"fold_{fold_idx}"]['train']['mlogloss'])
        all_val.append(evals_result_dict[f"fold_{fold_idx}"]['val']['mlogloss'])
        all_test.append(evals_result_dict[f"fold_{fold_idx}"]['test']['mlogloss'])

    plot_train_val_losses(
        all_train + all_val, [f'train fold {i}' for i in range(len(all_train))]+[f'val fold {i}' for i in range(len(all_val))],
        'train_val_losses_vs_epoch', plot_dirpath, 
        linestyles=['solid']*len(all_train) + ['dashed']*len(all_val),
    )
    plot_train_val_losses(
        all_train + all_test, [f'train fold {i}' for i in range(len(all_train))]+[f'test fold {i}' for i in range(len(all_test))],
        'train_test_losses_vs_epoch', plot_dirpath,
        linestyles=['solid']*len(all_train) + ['dotted']*len(all_test),
    )
    avg_train, avg_val, avg_test = np.mean(pad_list(all_train), axis=0), np.mean(pad_list(all_val), axis=0), np.mean(pad_list(all_test), axis=0)
    std_train, std_val, std_test = np.std(pad_list(all_train), axis=0), np.std(pad_list(all_val), axis=0), np.std(pad_list(all_test), axis=0)
    plot_train_val_losses(
        [avg_train, avg_val, avg_test], ['train avg', 'val avg', 'test avg'],
        'train_val_test_avg_vs_epoch', plot_dirpath,
        losses_std_arrs=[std_train, std_val, std_test],
        linestyles=['solid', 'dashed', 'dotted'],
    )

if __name__ == "__main__":
    make_losses()