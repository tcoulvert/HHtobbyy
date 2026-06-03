import json
import os

import numpy as np
import prettytable as pt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad

import hist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.integrate import trapezoid

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.workspace_utils import match_sample
from HHtobbyy.event_discrimination.models import map_model_to_Model


dfdataset_config = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/DFDatasets/vManos/24_Manos_2026-04-20_12-49-04/dataset_config.json"
dfdataset = DFDataset(dfdataset_config)
model, model_config = "MLP", {"output_dirpath": "/uscms/home/tsievert/nobackup/XHYbbgg/Model_Outputs/ManosMLP/2026-04-20_13-59-41", "activation_func": "ELU"}
model = map_model_to_Model(model)(dfdataset, model_config)

###################################################

test_pre = dfdataset.get_all_test(regex='*!Hto2G')

###################################################

dipho_mass_window = np.logical_and(test_pre['AUX_mass'].gt(100).to_numpy(), test_pre['AUX_mass'].lt(180).to_numpy())
pho_mva_cut = np.logical_and(test_pre['lead_mvaID'].gt(-0.7).to_numpy(), test_pre['sublead_mvaID'].gt(-0.7).to_numpy())
snt_cuts = np.logical_and(dipho_mass_window, pho_mva_cut)

test = test_pre.loc[snt_cuts]
# test = test_pre

if 'Run3_2025/data' in pd.unique(test['AUX_sample_era']):
    print('upscaling 2024 MC for 2025 data')
    test.loc[test['AUX_sample_era'].eq('Run3_2024/sim'), 'AUX_eventWeight'] = 2 * test.loc[test['AUX_sample_era'].eq('Run3_2024/sim'), 'AUX_eventWeight']
vbf23_mask = np.logical_and(
    test['AUX_sample_era'].isin(['Run3_2023/sim/preBPix', 'Run3_2023/sim/postBPix']),
    test['AUX_sample_name'].eq('VBFHH')
)  # makeup for lack of 2022 VBFHH signal
test.loc[vbf23_mask, 'AUX_eventWeight'] = 2.27 * test.loc[vbf23_mask, 'AUX_eventWeight']
# if '' in pd.unique(test['AUX_sample_era']):
#     print('upscaling 2022-23 MC for Run2 data')
#     run2_lumi = 138
#     run3_lumi = 270 if 'Run3_2025/data' in pd.unique(test['AUX_sample_era']) else 170
#     # run2_lumi_ratio = 2.23
#     run2_xs_reweight = {
#         # Signal #
#         'GluGluToHH': 0.75, 'VBFHH': 1,

#         # Resonant (Mgg) background #
#         # Fake b-jets
#         'GluGluHToGG': 0.93, 'VBFHToGG': 0.924, 'WmHToGG': 0.93, 'WpHToGG': 0.93,
#         # Real b-jets
#         'ttHToGG': 0.85, 'bbHToGG': 1,
#         # Resonant b-jets
#         'VHToGG': 0.93, 'ZHToGG': 0.93,

#         # Non-resonant (Mgg) background #
#         # Fake photons, fake b-jets
#         'GJet': 1,
#         # Fake photons
#         'TTG': 1,
#         # Fake b-jets
#         'GGJets': 0.98,
#         # Real b-jets
#         'TTGG': 1,

#         # Data-driven background #
#         'DDQCDGJets': 1
#     }
#     era_mask = test['AUX_sample_era'].isin(['Run3_2022/sim/preEE', 'Run3_2022/sim/postEE', 'Run3_2023/sim/preBPix', 'Run3_2023/sim/postBPix', 'Run3_2024/sim'])
#     # era_mask = test['AUX_sample_era'].isin(['Run3_2022/sim/preEE', 'Run3_2022/sim/postEE', 'Run3_2023/sim/preBPix', 'Run3_2023/sim/postBPix'])
#     for sample_name in pd.unique(test['AUX_sample_name']):
#         if sample_name not in run2_xs_reweight: continue
#         sample_mask = np.logical_and(era_mask, test[f'AUX_sample_name'].eq(sample_name))
#         print(f"{sample_name} yield before upscale: {test.loc[sample_mask, 'AUX_eventWeight'].sum()}")
#         test.loc[sample_mask, 'AUX_eventWeight'] = (run2_lumi * run2_xs_reweight[sample_name] + run3_lumi) / run3_lumi * test.loc[sample_mask, 'AUX_eventWeight']
#         print(f"{sample_name} yield after upscale: {test.loc[sample_mask, 'AUX_eventWeight'].sum()}")

###################################################

test['lumi:event'] = test['AUX_lumi'].astype(int).astype(str) + np.array([':']*len(test)) + test['AUX_event'].astype(int).astype(str)

with open('/uscms/home/tsievert/nobackup/XHYbbgg/HHtobbyy/src/HHtobbyy/misc/snt_transfer/boosted_events.json', 'r') as f:
    boosted_dict = json.load(f)
    boosted_events = boosted_dict['categories']

test['is_boosted'] = np.zeros(len(test), dtype=bool)
for sample_names, ids in boosted_events.items():
    for sample_name in sample_names.split('*'):
        sample_mask = test['AUX_sample_name'].eq(sample_name).to_numpy()
        id_mask = test['lumi:event'].isin(ids).to_numpy()
        test.loc[np.logical_and(sample_mask, id_mask), 'is_boosted'] = True
        print(f"Num {sample_name} events passing boosted: {np.sum(np.logical_and(sample_mask, id_mask))}")

###################################################

oldcats = {
    # 'VBFHH': {
    #     "AUX_DVBFHH": 0.774,
    # },
    'cat1': {
        "AUX_DggFHH": 0.20105975753991212,
        "AUX_DnonRes": 0.00033920225777953755,
        "AUX_DRes": 0.5155782175840005
    },
    'cat2': {
        "AUX_DggFHH": 0.8633095100474404,
        "AUX_DnonRes": 0.0015707913189143552,
        "AUX_DRes": 0.7641090808766726
    },
    'cat3': {
        "AUX_DggFHH": 0.7121864823883823,
        "AUX_DnonRes": 0.006492482419944794,
        "AUX_DRes": 0.5593729774038129
    }
}

cats = {
    "cat0": {
        "is_boosted": 1
    },
    "cat1": {
        "AUX_DVBFHH": 0.818,  # >
    },
    "cat2": {
        "AUX_DggFHH": 0.82294471227980193,  # >
        "AUX_DnonRes": 0.00020721235210142,  # <
        "AUX_DRes": 0.88339185576323398,  # <
    },
    "cat3": {
        "AUX_DggFHH": 0.95411234656848931,  # >
        "AUX_DnonRes": 0.57467284921409301,  # <
        "AUX_DRes": 0.00931784785851750,  # <
    },
    "cat4" : {
        "AUX_DggFHH": 0.60161934781392234,  # >
        "AUX_DnonRes": 0.00190229357061479,  # <
        "AUX_DRes": 0.80095654321512555,  # <
    }
}

###################################################

#############################################################
# ASCii histogram for rapid plotting
def ascii_hist(x, bins=10, weights=None):
    N,X = np.histogram(x, bins=bins, weights=weights)
    width = 50
    nmax = np.max(N.max())
    for (xi, n) in zip(X,N):
        bar = '#'*int(n*width/nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi,bar))
def ascii_hist(x, bins=10, weights=None, fit=None):
    N,X = np.histogram(x, bins=bins, weights=weights)
    width = 50
    nmax = np.max([N.max(), fit.max()])
    if fit is None:
        for (xi, n) in zip(X,N):
            bar = '#'*int(n*width/nmax)
            xi = '{0: <8.4g}'.format(xi).ljust(10)
            print('{0}| {1}'.format(xi,bar))
    else:
        for (xi, n, fiti) in zip(X,N,fit):
            bar = '#'*int(n*width/nmax)
            if fiti > n: bar = bar + ' '*(int(fiti*width/nmax) - int(n*width/nmax)) + '+'
            else: bar = ''.join([bar[j] if j != int(fiti*width/nmax) else '+' for j in range(len(bar))])
            xi = '{0: <8.4g}'.format(xi).ljust(10)
            print('{0}| {1}'.format(xi,bar))

#############################################################
# Sideband fit functions for nonRes bkg estimaton
def exp_func(x, a, b):
    return a * np.exp(b * x)
def sd_hist(mass: np.ndarray, weight: np.ndarray, fit_bins: list[float]):
    return hist.Hist(
        hist.axis.Regular(int((fit_bins[1]-fit_bins[0])//fit_bins[2]), fit_bins[0], fit_bins[1], name="var", growth=False, underflow=False, overflow=False), 
        storage='weight'
    ).fill(var=mass, weight=weight)
def exp_mass_fit(mass: np.ndarray, weight: np.ndarray, fit_bins: list[float], sigma: bool=False):
    _hist_ = sd_hist(mass, weight, fit_bins)
    params, _ = curve_fit(
        exp_func, _hist_.axes.centers[0]-_hist_.axes.centers[0][0], _hist_.values(), p0=(_hist_.values()[0], -0.1), 
        sigma=np.where(_hist_.values() != 0, np.sqrt(_hist_.variances()), 0.76) if sigma else None
    )
    print(_hist_)
    ascii_hist(mass, bins=np.arange(fit_bins[0], fit_bins[1], fit_bins[2]), weights=weight, fit=exp_func(_hist_.axes.centers[0]-_hist_.axes.centers[0][0], a=params[0], b=params[1]))
    return _hist_, params
def est_yield(mass: np.ndarray, weight: np.ndarray, fit_bins: list[float], sr_masscut: list[float], sigma: bool=False):
    _hist_, fit_params = exp_mass_fit(mass, weight, fit_bins, sigma=sigma)
    return quad(exp_func, sr_masscut[0]-_hist_.axes.centers[0][0], sr_masscut[1]-_hist_.axes.centers[0][0], args=tuple(fit_params))[0] / fit_bins[2]

###################################################

data_samples = [sample_name for sample_name in sorted(pd.unique(test['AUX_sample_name']).tolist()) if match_sample(sample_name, ['Data']) is not None]
sideband_samples = [sample_name for sample_name in sorted(pd.unique(test['AUX_sample_name']).tolist()) if match_sample(sample_name, ['GJet', 'TTG']) is not None]
non_sideband_samples = [sample_name for sample_name in sorted(pd.unique(test['AUX_sample_name']).tolist()) if sample_name not in data_samples+sideband_samples]
print(f"All samples: {sorted(pd.unique(test['AUX_sample_name']).tolist())}")
print(f"Data samples: {data_samples}")
print(f"nonRes MC samples: {sideband_samples}")
print(f"Res/Signal MC samples: {non_sideband_samples}")


# field_names = ['Category'] + non_sideband_samples + ['nonRes SB fit', 'data SB fit', 's/b with nonRes fit', 's/b with data fit']
# field_names = ['Category'] + non_sideband_samples + ['nonRes SB fit', 'data SB fit']
field_names = ['Category'] + non_sideband_samples + sideband_samples
table = pt.PrettyTable(field_names=field_names, float_format=".5")

not_prev_cut_mask = {}
for i, (name, cuts) in enumerate(cats.items()):

    for era in ['Run3_2022/sim/preEE', 'Run3_2022/sim/postEE', 'Run3_2023/sim/preBPix', 'Run3_2023/sim/postBPix', 'Run3_2024/sim', 'Run2']:
        if era == 'Run2':
            era_mask = test['AUX_sample_era'].eq('Run3_2022/sim/preEE'); evtwt_factor = (138 / 7.99)
        else:
            era_mask = test['AUX_sample_era'].eq(era); evtwt_factor = 1
        new_row = [name+' '+''.join(era.split('_')[-1].replace('/sim', '').split('/'))]
        nonRes_sideband = pd.DataFrame({'mass': pd.Series(dtype='float'), 'weight_tot': pd.Series(dtype='float')})

        for sample in non_sideband_samples+sideband_samples:

            sample_mask = np.logical_and(era_mask, test['AUX_sample_name'].eq(sample))
            if sample not in not_prev_cut_mask: not_prev_cut_mask[sample+era] = sample_mask

            pass_cut_mask = not_prev_cut_mask[sample+era]
            if i != 0:
                pass_cut_mask = np.logical_and(
                    pass_cut_mask, np.logical_and(test.loc[:, 'AUX_nonResReg_vbfpair_dijet_mass_DNNreg'].gt(80), test.loc[:, 'AUX_nonResReg_vbfpair_dijet_mass_DNNreg'].lt(190))
                )
            for cut_name, cut in cuts.items():
                pass_cut_mask = np.logical_and(
                    pass_cut_mask, test.loc[:, cut_name].gt(cut).to_numpy() 
                    if 'HH' in cut_name else (
                        test.loc[:, cut_name].lt(cut).to_numpy() 
                        if 'D' in cut_name else 
                        test.loc[:, cut_name].eq(cut).to_numpy()
                    )
                )
            pass_cut_sr_mask = np.logical_and(
                pass_cut_mask,
                np.logical_and(test.loc[:, 'AUX_mass'].gt(115.).to_numpy(), test.loc[:, 'AUX_mass'].lt(135.).to_numpy())
            )
            new_row.append((evtwt_factor * test.loc[pass_cut_sr_mask, 'AUX_eventWeight']).sum())

            not_prev_cut_mask[sample+era] = np.logical_and(not_prev_cut_mask[sample+era], ~pass_cut_mask)

        # for sb_sample_name, sb_sample_arr in zip(['data', 'nonRes'], [data_samples, sideband_samples]):
        #     sb_mask = np.zeros(len(test), dtype=bool)
        #     for sample in sb_sample_arr: sb_mask = np.logical_or(sb_mask, test['AUX_sample_name'].eq(sample))
        #     if sb_sample_name not in not_prev_cut_mask: not_prev_cut_mask[sb_sample_name] = sb_mask

        #     pass_cut_mask = not_prev_cut_mask[sb_sample_name]
        #     if i != 0:
        #         pass_cut_mask = np.logical_and(
        #             pass_cut_mask, np.logical_and(test.loc[:, 'AUX_nonResReg_vbfpair_dijet_mass_DNNreg'].gt(80), test.loc[:, 'AUX_nonResReg_vbfpair_dijet_mass_DNNreg'].lt(190))
        #         )
        #     for cut_name, cut in cuts.items():
        #         pass_cut_mask = np.logical_and(
        #             pass_cut_mask, test.loc[:, cut_name].gt(cut).to_numpy() 
        #             if 'HH' in cut_name else (
        #                 test.loc[:, cut_name].lt(cut).to_numpy() 
        #                 if 'D' in cut_name else 
        #                 test.loc[:, cut_name].eq(cut).to_numpy()
        #             )
        #         )
        #     pass_cut_sb_mask = np.logical_and(
        #         pass_cut_mask,
        #         np.logical_or(test.loc[:, 'AUX_mass'].lt(120.).to_numpy(), test.loc[:, 'AUX_mass'].gt(130.).to_numpy())
        #     )
        #     if np.sum(pass_cut_sb_mask) != 0:
        #         sb_est_yield = est_yield(test.loc[pass_cut_sb_mask, 'AUX_mass'], test.loc[pass_cut_sb_mask, 'AUX_eventWeight'], [100., 180., 5.], [122.5, 127.])
        #     else:
        #         sb_est_yield = 0
        #     new_row.append(sb_est_yield)

        # sum_singleH = new_row[1] + sum(new_row[4:11])
        # new_row.append(new_row[2] / (sum_singleH + new_row[11]))
        # new_row.append(new_row[2] / (sum_singleH + new_row[12]))

        table.add_row(new_row)

print(table)
