# Stdlib packages
import argparse
import logging
import os
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
sys.path.append(os.path.join(GIT_REPO, "training/"))
sys.path.append(os.path.join(GIT_REPO, "evaluation/"))

# Module packages
from plotting_utils import (
    plot_filepath, make_plot_data, combine_prepostfix
)
from training_utils import get_dataset_dirpath
from evaluation_utils import transform_preds_options

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
    "--bins", 
    type=int,
    default=1000,
    help="Number of bins to use for histogram"
)
parser.add_argument(
    "--discriminator", 
    choices=transform_preds_options(),
    default=transform_preds_options()[0],
    help="Defines the discriminator to use for output scores, discriminators are implemented in evaluation_utils"
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
DENSITY = args.density
LOGY = args.logy
BINS = args.bins
DISCRIMINATOR = args.discriminator
PLOT_TYPE = f'output_{DISCRIMINATOR}' + '_unweighted' if not WEIGHTS else '_MCweighted'

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_output_scores(
    plot_data: dict, plot_dirpath: str, pred_idx: int,
    plot_prefix: str='', plot_postfix: str=''
):
    if WEIGHTS:
        plot_postfix = combine_prepostfix(plot_postfix, 'MCweight', fixtype='postfix')
    if DENSITY:
        plot_postfix = combine_prepostfix(plot_postfix, 'density', fixtype='postfix')
        
    plt.figure(figsize=(9,7))

    hist_axis = hist.axis.Regular(BINS, 0., 1., name='var', growth=False, underflow=False, overflow=False)
    hists, labels = [], []
    for class_name, class_data in plot_data.items():
        hists.append(
            hist.Hist(hist_axis, storage='weight').fill(
                var=class_data['preds'], 
                weight=class_data['weights'] if WEIGHTS else np.ones_like(class_data['preds'])
            )
        )
        labels.append(class_name)

    hep.histplot(
        hists, yerr=WEIGHTS, alpha=0.8, density=DENSITY, histtype='step', label=labels
    )

    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Output score')
    if LOGY:
        plt.yscale('log')
    
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()


if __name__ == "__main__":
    make_plot_data(TRAINING_DIRPATH, DATASET_DIRPATH, args.dataset, args.discriminator, PLOT_TYPE, plot_output_scores, project_1D=True)