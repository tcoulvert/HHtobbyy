# Stdlib packages
import os
import subprocess

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

# Module packages
from plotting_utils import (
    make_plot_dirpath, plot_filepath
)

################################


CWD = os.getcwd()
FILL_VALUE = -999

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_1dhist(
    np_arrs, training_dirpath: str, plot_type: str, var_label: str,
    weights=None,
    subplots: tuple = None, stack: bool=False,
    labels: list=None, plot_prefix: str=None,  plot_postfix: str=None
):
    close = not (subplots is None)
    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)

    if len(np.shape(np_arrs)) == 1: np_arrs = [np_arrs]
    elif len(np.shape(np_arrs)) > 2: raise IndexError(f"Input array should be either a 1D numpy array or a list of 1D numpy arrays, your shape is {np.shape(np_arr)}")
    
    hists = []
    for i, arr in enumerate(np_arrs):
        var_mask = (
            (arr != FILL_VALUE)
            & np.isfinite(arr)
        )
        good_var_bool = np.any(var_mask) and np.min(arr[var_mask]) != np.max(arr[var_mask])

        max_val = np.max(arr[var_mask]) if good_var_bool else 0.
        min_val = np.min(arr[var_mask]) if good_var_bool else 1.

        var_hist = hist.Hist(
            hist.axis.Regular(100, min_val, max_val, name="var", label=var_label, growth=True), 
            storage='weight'
        ).fill(var=arr[var_mask], weight=weights[i] if weights is not None else np.ones_like(arr[var_mask]))
        hists.append(var_hist)

    if subplots is None: fig, ax = plt.subplots()
    else: fig, ax = subplots

    hep.cms.lumitext(f"Run3" + r" (13.6 TeV)", ax=ax)
    hep.cms.text("Simulation", ax=ax)

    if stack:
        hep.histplot(hists, ax=ax, histtype="step", yerr=True, label=labels, stack=stack)
    else:
        for hist, label in zip(hists, labels): hep.histplot(hists, ax=ax, histtype="step", yerr=True, label=label)
    plt.legend()
    plt.yscale('log')

    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix), 
        bbox_inches='tight'
    )
    plt.savefig(
        plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
        bbox_inches='tight'
    )
    if close: plt.close()

