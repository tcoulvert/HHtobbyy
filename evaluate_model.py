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
from data_processing_BDT import process_data

gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

# lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v3/"
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v3.1/"
Run3_2022 = 'Run3_2022_mergedFullResolved/sim'
Run3_2023 = 'Run3_2023_mergedFullResolved/sim'
Run3_2024 = 'Run3_2024_mergedFullResolved/sim'

def get_filepath_dict(syst_name: str='nominal'):
    return {
        'ggF HH': [
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", # central v2 preEE name
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",  # central v2 postEE name
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",  # thomas name
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",

            # lpc_fileprefix+Run3_2022+f"/preEE/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet", 
            # lpc_fileprefix+Run3_2022+f"/postEE/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet",
            # lpc_fileprefix+Run3_2023+f"/preBPix/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet", 
            # lpc_fileprefix+Run3_2023+f"/postBPix/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet",

            # kappa lambda scan #
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
        ],
        'ttH + bbH': [
            # ttH
            lpc_fileprefix+Run3_2022+f"/preEE/ttHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/ttHToGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/ttHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/ttHtoGG/{syst_name}/*merged.parquet",
            # bbH
            lpc_fileprefix+Run3_2022+f"/preEE/bbHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/bbHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/bbHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/bbHtoGG/{syst_name}/*merged.parquet",
        ],
        'VH': [
            # VH
            lpc_fileprefix+Run3_2022+f"/preEE/VHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/VHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/VHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/VHtoGG/{syst_name}/*merged.parquet",
            # ZH
            lpc_fileprefix+Run3_2022+f"/preEE/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet",
            # W-H
            lpc_fileprefix+Run3_2022+f"/preEE/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            # W+H
            lpc_fileprefix+Run3_2022+f"/preEE/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
        ],
        'non-res + ggFH + VBFH': [
            # GG + 3Jets 40-80
            lpc_fileprefix+Run3_2022+f"/preEE/GGJets_MGG-40to80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GGJets_MGG-40to80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GGJets_MGG-40to80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GGJets_MGG-40to80/{syst_name}/*merged.parquet",
            # GG + 3Jets 80-
            lpc_fileprefix+Run3_2022+f"/preEE/GGJets_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GGJets_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GGJets_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GGJets_MGG-80/{syst_name}/*merged.parquet",
            # GJet pT 20-40
            lpc_fileprefix+Run3_2022+f"/preEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            # GJet pT 40-inf
            lpc_fileprefix+Run3_2022+f"/preEE/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            # ggF H
            lpc_fileprefix+Run3_2022+f"/preEE/GluGluHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGluHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGluHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGluHtoGG/{syst_name}/*merged.parquet",
            # VBF H
            lpc_fileprefix+Run3_2022+f"/preEE/VBFHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/VBFHToGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/VBFHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/VBFHtoGG/{syst_name}/*merged.parquet",
        ],
    }

FILEPATHS_DICT = get_filepath_dict()

MODEL_FILEPATH = os.path.join(
    '/uscms/home/tsievert/nobackup/XHYbbgg/HHtobbyy/MultiClassBDT_model_outputs/v14/v3_vars_DijetMass_22_23/2025-07-08_15-47-07',
    ''
)
MOD_VALS = (5, 5)

order = ['ggF HH', 'ttH + bbH', 'VH', 'non-res + ggFH + VBFH']

param = {}

# Booster parameters #

# v12 #
param['eta']              = 0.05 # learning rate
num_trees = round(25 / param['eta'])  # number of trees to make
param['max_depth']        = 10  # maximum depth of a tree
param['subsample']        = 0.2 # fraction of events to train tree on
param['colsample_bytree'] = 0.6 # fraction of features to train tree on
param['num_class']        = len(order) # num classes for multi-class training
param['device']           = 'cuda'
param['tree_method']      = 'gpu_hist'
param['max_bin']          = 512
param['grow_policy']      = 'lossguide'
param['sampling_method']  = 'gradient_based'
param['min_child_weight'] = 0.25


# Learning task parameters
param['objective']   = 'multi:softprob'   # objective function
param['eval_metric'] = 'merror'
param = list(param.items()) + [('eval_metric', 'mlogloss')]  # evaluation metric for cross validation

# custom eval_metrics
def one_hot_encoding(cat_labels: np.ndarray):
    one_hot = np.zeros((np.shape(cat_labels)[0], np.max(cat_labels)+1))
    for i in range(np.max(cat_labels)):
        one_hot[:, i] = (cat_labels == i)
    return one_hot

def mlogloss_binlogloss(
    predt: np.ndarray, dtrain: xgb.DMatrix, mLL=True, **kwargs
):
    assert (len(kwargs) == 0 and mLL) or len(kwargs) == (len(order) - 1)

    mweight = dtrain.get_weight()
    monehot = one_hot_encoding(dtrain.get_label())
    mlogloss = log_loss(monehot, predt, sample_weight=mweight, normalize=False)

    bkgloglosses = {}
    for i, (key, value) in enumerate(kwargs.items(), start=1):
        bkgbool = np.logical_or(mweight == 0, mweight == i)
        bkgloglosses[key] = value * log_loss(
            monehot[bkgbool], predt[bkgbool, 0] / (predt[bkgbool, 0] + predt[bkgbool, i]),
            sample_weight=mweight[bkgbool], normalize=False
        )

    if len(bkgloglosses) > 0 and mLL:
        return f'mLL+binLL@{bkgloglosses.keys()}', float(np.sum([mlogloss]+list(bkgloglosses.values())))
    elif len(bkgloglosses) > 0:
        return f'binLL@{bkgloglosses.keys()}', float(np.sum(bkgloglosses.values()))
    else:
        return 'mLL', float(mlogloss)

def thresholded_weighted_merror(predt: np.ndarray, dtrain: xgb.DMatrix, threshold=0.95):
    """Used when there's no custom objective."""
    # No need to do transform, XGBoost handles it internally.
    weights = dtrain.get_weight()
    thresh_weight_merror = np.where(
        np.logical_and(
            np.max(predt, axis=1) >= threshold,
            np.argmax(predt, axis=1) == dtrain.get_label()
        ),
        0,
        weights
    )
    return f'WeightedMError@{threshold:.2f}', np.sum(thresh_weight_merror)

