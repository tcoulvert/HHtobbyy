# %matplotlib widget
# Stdlib packages
import copy
import datetime
import glob
import json
import os
import re
import warnings
from pathlib import Path

# Common Py packages
import awkward as ak
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.special import logit as inverse_sigmoid

# HEP packages
import gpustat
import h5py
import hist
import mplhep as hep
import xgboost as xgb
from cycler import cycler

# ML packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
from sklearn.metrics import log_loss
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Module packages
from plotting_utils import (
    plot_filepath,
)

################################


gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def plot_output_scores(
    sigs_and_bkgs, order, plot_name, plot_dirpath,
    plot_prefix='', plot_postfix='', bins=1000, weights=None, log=False, arctanh=False
):
    plt.figure(figsize=(9,7))

    if arctanh:
        end_point = 6.
    else:
        end_point = 1.
    hist_axis = hist.axis.Regular(bins, 0., end_point, name='var', growth=False, underflow=False, overflow=False)
    hists, labels = [], []
    for sample_name in order:
        if sample_name not in sigs_and_bkgs:
            continue
        hists.append(
            hist.Hist(hist_axis, storage='weight').fill(
                var=sigs_and_bkgs[sample_name], 
                weight=weights[sample_name] if weights is not None else np.ones_like(sigs_and_bkgs[sample_name])
            )
        )
        labels.append(sample_name)
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

