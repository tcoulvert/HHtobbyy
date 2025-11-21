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
    plot_filepath, make_plot_dirpath
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
parser.add_argument(
    "--importance_type",
    choices=['weight', 'gain', 'cover', 'total_gain', 'total_cover'],
    default='weight',
    help="Method to calculate feature importance"
)
parser.add_argument(
    "--logy", 
    action="store_true",
    help="Boolean to make plots log-scale on y-axis"
)

################################


CWD = os.getcwd()
args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
DATASET_DIRPATH = get_dataset_dirpath(TRAINING_DIRPATH)
IMPORTANCE_TYPE = args.importance_type
LOGY = args.logy
PLOT_TYPE = 'feature_importance'

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_feature_importance(
    feature_scores, feature_names, plot_dirpath, 
    plot_prefix='', plot_postfix='', 
):
    plt.figure(figsize=(18,14))

    plt.barh(
        np.arange(len(feature_scores)), feature_scores, align='center'
    )
    plt.yticks(np.arange(len(feature_scores)), feature_names, fontsize=8)
    plt.ylabel('Features')
    plt.xlabel(f'F score ({IMPORTANCE_TYPE})')
    if LOGY: plt.xscale('log')
    
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()


def make_feature_importance():
    plot_dirpath = make_plot_dirpath(TRAINING_DIRPATH, PLOT_TYPE)

    get_booster = get_model_func(TRAINING_DIRPATH)
    
    stdjson_filepath = os.path.join(DATASET_DIRPATH, 'standardization.json')
    with open (stdjson_filepath, 'r') as f:
        stdjson = json.load(f)
    feature_names = sorted(stdjson['cols'])

    full_score_dict = None
    for fold_idx in range(get_n_folds(DATASET_DIRPATH)):
        booster = get_booster(fold_idx)
        booster.feature_names = feature_names

        score_dict = booster.get_score(importance_type=IMPORTANCE_TYPE)
        if fold_idx == 0:
            full_score_dict = copy.deepcopy(score_dict)
        else:
            for feature_name in full_score_dict.keys():
                full_score_dict[feature_name] += score_dict[feature_name]

        scores = np.array([score for score in score_dict.values()])

        sorted_feature_names = np.array(feature_names)[np.argsort(scores)]
        sorted_scores = np.sort(scores)

        plot_feature_importance(
            sorted_scores, sorted_feature_names, plot_dirpath, 
            plot_prefix=IMPORTANCE_TYPE, plot_postfix=f"{fold_idx}"
        )

    full_scores = np.array([score / get_n_folds(DATASET_DIRPATH) for score in score_dict.values()])
    sorted_feature_names = np.array(feature_names)[np.argsort(full_scores)]
    sorted_full_scores = np.sort(sorted_full_scores)

    plot_feature_importance(
        sorted_full_scores, sorted_feature_names, plot_dirpath, 
        plot_prefix=IMPORTANCE_TYPE
    )


if __name__ == "__main__":
    make_feature_importance()