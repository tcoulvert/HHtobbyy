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
import pandas as pd
import prettytable as pt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# HEP packages
import hist

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.Categorization import CategorizationConfig
from HHtobbyy.Categorization.categorization_utils import *

################################

class Categorization:
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        self.catconfig = CategorizationConfig(dfdataset, config)

    def apply_cut(self, df: pd.DataFrame, cut: list, anti: bool=False):
        cut_mask = np.ones(len(df), dtype=bool)
        for cut_, name, cutdir in zip(cut, self.catconfig.transform_names, self.catconfig.cutdir):
            if '>' in cutdir: cut_mask = np.logical_and(cut_mask, df[name].gt(cut_))
            else: cut_mask = np.logical_and(cut_mask, df[name].lt(cut_))
        if anti: return ~cut_mask
        else: return cut_mask

    def run(self, model: Model):
        MCsignal = self.dfdataset.get_all_test(regex=self.catconfig.signal_samples)
        MCres = self.dfdataset.get_all_test(regex=self.catconfig.res_samples)
        MCnonRes = self.dfdataset.get_all_test(regex=self.catconfig.nonres_samples)
        Data = self.dfdataset.get_all_test(regex='Data')

        MC_names = sorted(pd.unique(MCres.loc[:,f"{self.dfdataset.aux_var_prefix}sample_name"].tolist()))
        field_names = ['Category', 'FoM (s/b)'] + MC_names + ['nonRes MC -- SB fit', 'Data -- SB fit']
        table = pt.PrettyTable(field_names=field_names)

        def get_prev_cuts(cats_: dict):
            if len(cats_) == 0: return None
            prev_cuts = []
            for cat_ in cats_.values(): prev_cuts.apend(cat_['cut'])
            return prev_cuts
        
        def slim_df(df: pd.DataFrame, mask: np.ndarray, col_map: dict):
            return df.loc[mask, col_map.keys()].rename(col_map)

        cats = {}
        signal_mask, res_mask, nonRes_mask, data_mask = (
            np.ones(len(MCsignal), dtype=bool), np.ones(len(MCres), dtype=bool), 
            np.ones(len(MCnonRes), dtype=bool), np.ones(len(Data), dtype=bool)
        )
        for cat_idx in range(1, self.catconfig.n_cats+1):
            prev_cut = get_prev_cuts(cats)
            best_fom, best_cut = self.catconfig.get_catmethod()(
                slim_df(MCsignal, signal_mask, self.catconfig.opt_columns_map),
                slim_df(MCres, res_mask, self.catconfig.opt_columns_map),
                slim_df(MCnonRes, nonRes_mask, self.catconfig.opt_columns_map),
                slim_df(Data, data_mask, self.catconfig.opt_columns_map),
                self.catconfig,
                prev_cut
            )

            best_evals = {
                name: np.sum(slim_df(MCres, res_mask, self.catconfig.opt_columns_map).loc[np.logical_and(self.apply_cut(MCres), mass_cut(MCres.rename({f})))
            }

            cats[f'cat{cat_idx}'] = {
                'fom': best_fom.item(), 'cut': best_cut.tolist(), 'evals': best_evals.tolist()
            }

            signal_mask, res_mask, nonRes_mask, data_mask = (
                self.apply_cut(MCsignal, best_cut, anti=True), 
                self.apply_cut(MCres, best_cut, anti=True),
                self.apply_cut(MCnonRes, best_cut, anti=True), 
                self.apply_cut(Data, best_cut, anti=True),
            )

            new_row = [f'Merged folds - Cat {cat_idx}', best_fom] + [best_evals[name] for name in MC_names]
            table.add_row(new_row)