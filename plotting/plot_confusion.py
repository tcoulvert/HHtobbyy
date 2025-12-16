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
from cycler import cycler

# ML packages
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score

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
    plot_filepath, make_plot_data, combine_prepostfix,
    float_to_str
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
    "--normalize", 
    choices=["true", "pred", "all"], 
    default="true",
    help="Boolean to nomralize confusion matrix scores"
)
parser.add_argument(
    "--beta", 
    type=float, 
    default=1.,
    help="Beta value to compute Fβ score with"
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
NORMALIZE = args.normalize
BETA = args.beta
PLOT_TYPE = 'confusion_matrix' + ('_unweighted' if not WEIGHTS else '_MCweighted')

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_confusion_matrix(
    conf_matrix, display_labels, fbeta, plot_dirpath: str,
    plot_prefix: str='', plot_postfix: str=''
):
    if WEIGHTS:
        plot_postfix = combine_prepostfix(plot_postfix, 'MCweight', fixtype='postfix')
    plot_postfix = combine_prepostfix(plot_postfix, f'NORM{NORMALIZE}_BETA{float_to_str(BETA)}', fixtype='postfix')

    plt.figure(figsize=(9,7))

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
    disp.plot(im_kw={'norm': 'log'})
    plt.title(f"Confusion Matrix with Binary signal Fβ score of {fbeta}")

    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(PLOT_TYPE, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    plt.close()


def make_confusion_matrix(
    plot_data: dict, transform_labels: list, plot_dirpath: str,
    plot_prefix: str='', plot_postfix: str=''
):
    conf_data = None
    for j, (class_name, class_data) in enumerate(plot_data.items()):
        if j == 0:
            conf_data = {
                data_name: np.concatenate([_class_data_[data_name] for _class_data_ in plot_data.values()])
                for data_name in class_data.keys()
            }
    conf_data['preds'] = np.argmax(conf_data['preds'], axis=1)

    conf_matrix = confusion_matrix(
        conf_data['labels'], conf_data['preds'],
        sample_weight=conf_data['weights'] if WEIGHTS else None, normalize=NORMALIZE
    )
    fbeta = fbeta_score(
        conf_data['labels'], conf_data['preds'], pos_label=0,
        beta=BETA, sample_weight=conf_data['weights'] if WEIGHTS else None
    )

    plot_confusion_matrix(
        conf_matrix, [class_name for class_name in plot_data.keys()], fbeta, plot_dirpath,
        plot_prefix=plot_prefix, plot_postfix=plot_postfix
    )


if __name__ == "__main__":
    make_plot_data(TRAINING_DIRPATH, DATASET_DIRPATH, args.dataset, transform_preds_options()[0], PLOT_TYPE, make_confusion_matrix)