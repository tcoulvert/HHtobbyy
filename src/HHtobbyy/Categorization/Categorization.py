# Stdlib packages
import os

# Common Py packages
import numpy as np
import pandas as pd
import prettytable as pt

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.evaluation import class_discriminator_columns
from .CategorizationConfig import CategorizationConfig
from .categorization_utils import *

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

    def get_sr_cut_mask(self, df: pd.DataFrame, cut: list):
        return np.logical_and(
            self.apply_cut(df, cut), 
            mass_cut(df, self.catconfig.SR_masscut, self.dfdataset.aux_var_prefix)
        )
    def get_yield_from_cut(self, df: pd.DataFrame, cut: list):
        sr_cut_mask = self.get_sr_cut_mask(df, cut)
        return df.loc[sr_cut_mask, f"{self.dfdataset.aux_var_prefix}eventWeight"].sum()
    
    def get_opt_df(self, df: pd.DataFrame):
        disc_columns = class_discriminator_columns(self.dfdataset.class_sample_map.keys())
        nD_predictions = df[disc_columns].to_numpy(copy=True)
        trns_predictions = self.catconfig.transform_func(nD_predictions)
        for i, trans_name in enumerate(self.catconfig.transform_names):
            df[trans_name] = trns_predictions[:, i]
        return df

    def run(self):
        MCsignal = self.get_opt_df(self.dfdataset.get_all_test(regex=['preEE*'+ sampl for sampl in self.catconfig.signal_samples]))
        MCres = self.get_opt_df(self.dfdataset.get_all_test(regex=['preEE*'+ sampl for sampl in self.catconfig.res_samples]))
        MCnonRes = self.get_opt_df(self.dfdataset.get_all_test(regex=['preEE*'+ sampl for sampl in self.catconfig.nonres_samples]))
        Data = self.get_opt_df(self.dfdataset.get_all_test(regex='2022*Data'))

        MC_names = sorted(pd.unique(MCres.loc[:,f"{self.dfdataset.aux_var_prefix}sample_name"]).tolist())
        field_names = ['Category', 'FoM (s/b)'] + MC_names + ['nonRes MC -- SB fit', 'Data -- SB fit']
        table = pt.PrettyTable(field_names=field_names)

        def get_prev_cuts(cats_: dict):
            if len(cats_) == 0: return None
            return [cat_['cut'] for cat_ in cats_.values()]

        cats = {}
        signal_mask, res_mask, nonRes_mask, data_mask = (
            np.ones(len(MCsignal), dtype=bool), np.ones(len(MCres), dtype=bool), 
            np.ones(len(MCnonRes), dtype=bool), np.ones(len(Data), dtype=bool)
        )
        for cat_idx in range(1, self.catconfig.n_cats+1):
            prev_cut = get_prev_cuts(cats)
            best_fom, best_cut = self.catconfig.get_catmethod()(
                MCsignal.loc[signal_mask], MCres.loc[res_mask], MCnonRes.loc[nonRes_mask],
                self.catconfig, prev_cut
            )

            best_evals = {
                **{
                    name: self.get_yield_from_cut(
                        MCres.loc[
                            np.logical_and(res_mask, MCres[f"{self.dfdataset.aux_var_prefix}sample_name"].eq(name))
                        ], best_cut
                    )
                    for name in pd.unique(MCres[f"{self.dfdataset.aux_var_prefix}sample_name"])
                },
                **{
                    name: est_yield(
                        df.loc[np.logical_and(mask, self.get_sr_cut_mask(df, best_cut)), f"{self.dfdataset.aux_var_prefix}mass"],
                        df.loc[np.logical_and(mask, self.get_sr_cut_mask(df, best_cut)), f"{self.dfdataset.aux_var_prefix}eventWeight"],
                        self.catconfig.fit_bins, self.catconfig.SR_masscut
                    )
                    for name, df, mask in zip(['nonRes MC -- SB fit', 'Data -- SB fit'], [MCnonRes, Data], [nonRes_mask, data_mask])
                }
            }

            cats[f'cat{cat_idx}'] = {
                'fom': best_fom.item(), 'cut': best_cut.tolist(), 'evals': best_evals
            }

            signal_mask, res_mask, nonRes_mask, data_mask = (
                self.apply_cut(MCsignal, best_cut, anti=True), self.apply_cut(MCres, best_cut, anti=True), 
                self.apply_cut(MCnonRes, best_cut, anti=True), self.apply_cut(Data, best_cut, anti=True),
            )

            new_row = [f'Merged folds - Cat {cat_idx}', best_fom] + [best_evals[name] for name in MC_names]
            table.add_row(new_row)

        print(table)
        eos.save_file_eos(cats, os.path.join(self.catconfig.output_dirpath, self.catconfig.cat_filename))