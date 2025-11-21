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
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))

# Module packages
from plotting_utils import (
    plot_filepath, combine_prepostfix
)
from BDT_preprocessing_utils import (
    log_standardize, apply_logs
)
from training_utils import get_dataset_dirpath

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

FILL_VALUE = -999

################################


CWD = os.getcwd()
args = parser.parse_args()
TRAINING_DIRPATH = os.path.join(args.training_dirpath, "")
if args.dataset_dirpath is None:
    DATASET_DIRPATH = get_dataset_dirpath(args.training_dirpath)
else:
    DATASET_DIRPATH = os.path.join(args.dataset_dirpath, '')
WEIGHTS = args.weights
SYST_NAME = args.syst_name
PLOT_TYPE = 'vars'

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_vars(df, output_dirpath, sample_name, title="pre-std, train0"):
    std_type, df_type = tuple(title.split(", "))
    plot_dirpath = os.path.join(output_dirpath, "plots", "_".join([std_type.replace('-', ''), df_type]))
    if not os.path.exists(plot_dirpath): os.makedirs(plot_dirpath)

    if "pre" in std_type: apply_logs(df)

    for var in df.columns:
        if log_standardize(var): var_label = f"ln({var})"
        else: var_label = var

        var_mask = (
            (df[var] != FILL_VALUE)
            & np.isfinite(df[var])
        )
        good_var_bool = np.any(var_mask) and np.min(df.loc[var_mask, var]) != np.max(df.loc[var_mask, var])

        max_val = np.max(df.loc[var_mask, var]) if good_var_bool else 0.
        min_val = np.min(df.loc[var_mask, var]) if good_var_bool else 1.

        var_hist = hist.Hist(
            hist.axis.Regular(100, min_val, max_val, name="var", label=var_label, growth=True), 
        ).fill(var=df.loc[var_mask, var])

        fig, ax = plt.subplots()
        hep.cms.lumitext(f"Run3" + r" (13.6 TeV)", ax=ax)
        hep.cms.text("Simulation", ax=ax)

        hep.histplot(var_hist, ax=ax, histtype="step", yerr=True, label=" - ".join([sample_name, title]))
        plt.legend()
        plt.yscale('log')

        plt.savefig(os.path.join(plot_dirpath, f"{var}_{std_type.replace('-', '')}_{df_type}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_dirpath, f"{var}_{std_type.replace('-', '')}_{df_type}.png"), bbox_inches='tight')
        plt.close()
        
if __name__ == "__main__":
    pass