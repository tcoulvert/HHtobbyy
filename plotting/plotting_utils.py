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

################################


gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


def pad_list(list_of_lists):
    max_length = np.max([len(list_i) for list_i in list_of_lists])
    for list_i in list_of_lists:
        while len(list_i) < max_length:
            list_i.append(list_i[-1])

    return list_of_lists

def plot_filepath(plot_name, plot_dirpath, plot_prefix, plot_postfix, format='png'):
    plot_prefix = plot_prefix + ('_' if plot_prefix != '' else '')
    plot_postfix = plot_postfix + ('_' if plot_postfix != '' else '')
    plot_name = plot_prefix + plot_name + plot_postfix + f'.{format}'

    plot_filepath = os.path.join(plot_dirpath, plot_name)
    return plot_filepath

def get_ttH_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])

def get_QCD_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])

OPTIMIZED_CUTS = {
    '1D': [0.9977, 0.9946, 0.9874],
    '2D': [
        [0.987, 0.9982],
        [0.92, 0.994],
        [0.92, 0.9864],
    ]
}
def cat_mask(score, index: int, EVAL_METHOD='2D'):

    if EVAL_METHOD == '1D':

        lower_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index]
        if index > 0:
            upper_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index - 1]
        else:
            upper_cut_value = 1.0
        
        mask = (
            score > lower_cut_value
        ) & (
            score <= upper_cut_value
        )

        return mask
    
    elif EVAL_METHOD == '2D':

        lower_ttH_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index][0]
        lower_QCD_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index][1]
        if index > 0:
            upper_QCD_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index - 1][1]
        else:
            upper_QCD_cut_value = 1.
        
        mask = (
            get_ttH_score(score) > lower_ttH_cut_value
        ) & (
            get_QCD_score(score) > lower_QCD_cut_value
        ) & (
            get_QCD_score(score) <= upper_QCD_cut_value
        )

        return mask
