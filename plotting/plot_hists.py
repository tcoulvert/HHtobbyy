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


def make_hists(arrs: list, var_label: str, weights: list=None):
    min_val, max_val = 0., 1.
    for i, arr in enumerate(arrs):
        var_mask = (
            (arr != FILL_VALUE)
            & np.isfinite(arr)
        )
        good_var_bool = np.any(var_mask) and np.min(arr[var_mask]) != np.max(arr[var_mask])
        min_val = np.min(arr[var_mask]) if good_var_bool and np.min(arr[var_mask]) < min_val else min_val
        max_val = np.max(arr[var_mask]) if good_var_bool and np.max(arr[var_mask]) > max_val else max_val

    hists = []
    for i, arr in enumerate(arrs):
        var_mask = (
            (arr != FILL_VALUE)
            & np.isfinite(arr)
        )

        var_hist = hist.Hist(
            hist.axis.Regular(50, min_val, max_val, name="var", label=var_label, growth=False, underflow=False), 
            storage='weight'
        ).fill(var=arr[var_mask], weight=weights[i][var_mask] if weights is not None else np.ones_like(arr[var_mask]))
        hists.append(var_hist)
    return hists


def plot_ratio(
    arrs1: list|np.ndarray, arrs2: list|np.ndarray, var_label: str, subplots: tuple,
    training_dirpath: str, plot_type: str,
    weights1: list|np.ndarray=None, weights2: list|np.ndarray=None, 
    yerr: bool=False, central_value=1.0, color='black', lw=2.,
    plot_prefix: str=None,  plot_postfix: str=None, save_and_close: bool=False
):
    """
    Does the ratio plot (code copied from Hist.plot_ratio_array b/c they don't
      do what we need.)
    """
    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)

    if type(arrs1) is np.ndarray: arrs1 = [arrs1]; weights1 = [weights1]
    if type(arrs2) is np.ndarray: arrs2 = [arrs2]; weights2 = [weights2]
    assert (
        len(np.shape(np.array(arrs1, dtype=object))) <= 2 
        and len(np.shape(np.array(arrs2, dtype=object))) <= 2
    ), f"Input arrays should be either a 1D numpy array or a list of 1D numpy arrays, your shapes are {np.shape(arrs1)} and {np.shape(arrs2)}"

    hists1 = make_hists(arrs1, var_label, weights=weights1)
    hists2 = make_hists(arrs2, var_label, weights=weights2)

    ratio = np.sum([hist1.values() for hist1 in hists1], axis=0) / np.sum([hist2.values() for hist2 in hists2], axis=0)
    print(f"median Data / MC for {var_label}: {np.median(ratio[np.isfinite(ratio)]):.2f}")
    numer_err, denom_err = np.sqrt(np.sum([hist1.variances() for hist1 in hists1], axis=0)) / np.sum([hist1.values() for hist1 in hists1], axis=0), np.sqrt(np.sum([hist2.variances() for hist2 in hists2], axis=0)) / np.sum([hist2.values() for hist2 in hists2], axis=0)
    for arr in [ratio, numer_err, denom_err]:  # Set 0 and inf to nan to hide during plotting
        arr[arr == 0] = np.nan  
        arr[np.isinf(arr)] = np.nan

    fig, axs = subplots[0], subplots[1]

    axs[1].set_ylim(0., 5.)
    axs[1].axhline(
        central_value, color="black", linestyle="solid", lw=1.
    )
    axs[1].stairs(
        ratio, edges=hists1[0].axes.edges[0], fill=False, 
        baseline=1., lw=lw, color=color, alpha=0.8
    )

    if yerr:
        axs[1].errorbar(
            hists1[0].axes.centers[0], ratio, yerr=numer_err, 
            fmt='none', lw=lw, color=color, alpha=0.8
        )
        axs[1].bar(
            hists1[0].axes.centers[0], height=denom_err * 2, width=(hists1[0].axes.centers[0][1] - hists1[0].axes.centers[0][0]), 
            bottom=(central_value - denom_err), color="green", alpha=0.5, hatch='//'
        )

    fig.subplots_adjust(hspace=0.05)
    axs[1].set_xlabel(axs[0].get_xlabel())
    axs[0].set_xlabel('')

    if save_and_close:
        plt.savefig(
            plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix), 
            bbox_inches='tight'
        )
        plt.savefig(
            plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
            bbox_inches='tight'
        )
        plt.close()

def ratio_error(numer_values, denom_values, numer_err, denom_err):
    ratio_err =  np.sqrt(
        np.power(denom_values, -2) * (
            np.power(numer_err, 2) + (
                np.power(numer_values / denom_values, 2) * np.power(denom_err, 2)
            )
        )
    )
    return ratio_err


def plot_1dhist(
    arrs: list|np.ndarray, training_dirpath: str, plot_type: str, var_label: str,
    weights: list|np.ndarray=None, yerr: bool=False, subplots: tuple = None, 
    histtype: str="step", stack: bool=False, labels: list|str=None, 
    logy: bool=False, density: bool=False,
    colors: list|str=None,
    plot_prefix: str='',  plot_postfix: str='', save_and_close: bool=False
):
    plot_dirpath = make_plot_dirpath(training_dirpath, plot_type)

    if type(arrs) is np.ndarray: arrs = [arrs]; weights = [weights]; labels = [labels]; colors = [colors]
    if len(np.shape(np.array(arrs, dtype=object))) > 2: raise IndexError(f"Input array should be either a 1D numpy array or a list of 1D numpy arrays, your shape is {np.shape(arrs)}")

    hists = make_hists(arrs, var_label, weights=weights)
    xs_order = np.argsort([np.sum(_hist_.values()) for _hist_ in hists])
    hists, weights = [hists[i] for i in xs_order], [weights[i] for i in xs_order]

    if subplots is None: fig, ax = plt.subplots()
    else: fig, ax = subplots

    hep.cms.lumitext(f"Run3" + r" (13.6 TeV)", ax=ax)
    hep.cms.text("Simulation", ax=ax)

    w2s = [None for _ in hists] if weights is None else [var_hist.variances() for var_hist in hists]
    labels = [None for _ in hists] if labels is None else [labels[i] for i in xs_order]
    colors = [None for _ in hists] if colors is None else [colors[i] for i in xs_order]

    if stack:
        hep.histplot(hists, ax=ax, histtype=histtype, yerr=np.sqrt(np.sum(w2s, axis=0)) if yerr else False, label=labels, stack=stack, density=density)
    else:
        for hist, w2, label, color in zip(hists, w2s, labels, colors): hep.histplot(hist, ax=ax, histtype=histtype, yerr=np.sqrt(w2) if yerr else False, label=label, density=density, color=color)
    
    ax.legend(bbox_to_anchor=(1, 1))
    if logy: ax.set_yscale('log')

    if save_and_close:
        plt.savefig(
            plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix), 
            bbox_inches='tight'
        )
        plt.savefig(
            plot_filepath(plot_type, plot_dirpath, plot_prefix, plot_postfix, format='pdf'), 
            bbox_inches='tight'
        )
        plt.close()

