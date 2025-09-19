# %matplotlib widget
# Stdlib packages
import glob
import os
import subprocess
import sys

# Common Py packages
import numpy as np

# HEP packages
import xgboost as xgb

# ML packages
from sklearn.metrics import log_loss

################################


GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))

from retrieval_utils import get_labelND

################################


def get_filepaths_func(base_filepath: str, syst_name: str='nominal'):
    return lambda fold_idx, dataset: {
        'ggF HH': glob.glob(os.path.join(base_filepath, "**", "*GluGlu*HH*kl-1p00*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True),
        'ttH + bbH': glob.glob(os.path.join(base_filepath, "**", "*ttH*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True) 
        + glob.glob(os.path.join(base_filepath, "**", "*bbH*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True),
        'VH': glob.glob(os.path.join(base_filepath, "**", "*VH*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*ZH*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*Wm*H*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*Wp*H*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True),
        'nonRes + ggFH + VBFH': glob.glob(os.path.join(base_filepath, "**", "*GGJets*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*GJet*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*GluGluH*GG*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True)
        + glob.glob(os.path.join(base_filepath, "**", "*VBFH*GG*", f"*{syst_name}*", f"*{dataset}{fold_idx}*.parquet"), recursive=True),
    }

def get_filepaths(base_filepath: str, fold_idx: int, dataset: str, syst_name: str='nominal'):
    """
    Returns the dictionary with the BDT classes as keys and the list of standardized dataset 
        filepaths as values.
    """
    print(get_filepaths_func(base_filepath, syst_name=syst_name))
    return get_filepaths_func(base_filepath, syst_name=syst_name)(fold_idx, dataset)


def mlogloss_binlogloss(
    predt: np.ndarray, dtrain: xgb.DMatrix, mLL=True, **kwargs
):
    # assert (len(kwargs) == 0 and mLL) or len(kwargs) == (n_classes - 1)
    assert (len(kwargs) == 0 and mLL)

    mweight = dtrain.get_weight()
    monehot = get_labelND(dtrain.get_label())
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