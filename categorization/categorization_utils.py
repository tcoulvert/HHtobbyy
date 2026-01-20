# Stdlib packages
import copy

# Common Py packages
import numba as nb
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

################################


def fom_mask(df: pd.DataFrame):
    return np.logical_and(df.loc[:, 'AUX_mass'].ge(122.5).to_numpy(), df.loc[:, 'AUX_mass'].le(127.).to_numpy())

def sideband_nonres_mask(df: pd.DataFrame):
    return np.logical_and(
        np.logical_or(df.loc[:, 'AUX_mass'].lt(120.).to_numpy(), df.loc[:, 'AUX_mass'].gt(130.).to_numpy()),
        np.logical_or(
            df.loc[:, 'AUX_sample_name'].eq('TTGG').to_numpy(),  # TTGG
            np.logical_or(
                np.logical_or(df.loc[:, 'AUX_sample_name'].eq('GJet').to_numpy(), df.loc[:, 'AUX_sample_name'].eq('GGJets').to_numpy()),  # GGJets or GJet
                df.loc[:, 'AUX_sample_name'].eq('DDQCDGJets').to_numpy()  # DDQCD GJet or GGJets
            )
        )
    )

@nb.njit
def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)

@nb.njit
def fom_s_over_b(s, b):
    return s / b



@nb.njit
def brute_force(
    signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
    bkg_sideband_scores, bkg_sideband_weights, 
    cuts, foms
):
    for i, cut in enumerate(cuts):
        signal_sr, bkg_sr, bkg_sideband = 0., 0., 0.
        for signal_sr_score, signal_sr_weight in zip(signal_sr_scores, signal_sr_weights):
            pass_cut_bool = True
            for score_j, cut_j in zip(signal_sr_score, cut): pass_cut_bool = pass_cut_bool & (score_j > cut_j)
            signal_sr += pass_cut_bool * signal_sr_weight
        for bkg_sr_score, bkg_sr_weight in zip(bkg_sr_scores, bkg_sr_weights):
            pass_cut_bool = True
            for score_j, cut_j in zip(bkg_sr_score, cut): pass_cut_bool = pass_cut_bool & (score_j > cut_j)
            bkg_sr += pass_cut_bool * bkg_sr_weight
        for bkg_sideband_score, bkg_sideband_weight in zip(bkg_sideband_scores, bkg_sideband_weights):
            pass_cut_bool = True
            for score_j, cut_j in zip(bkg_sideband_score, cut): pass_cut_bool = pass_cut_bool & (score_j > cut_j)
            bkg_sideband += pass_cut_bool * bkg_sideband_weight

        foms[i] = fom_s_over_b(signal_sr, bkg_sr) if bkg_sideband > 8. else 0.
    return foms

def grid_search(df: pd.DataFrame, cat_mask: np.ndarray, options_dict: dict):
    sr_mask = np.logical_and(cat_mask, fom_mask(df))
    sideband_mask = np.logical_and(cat_mask, sideband_nonres_mask(df))

    best_fom, best_cut = 0., [0. for _ in options_dict['TRANSFORM_COLUMNS']]

    all_foms, all_cuts = [], []

    # Signal events inside SR
    signal_sr_mask = np.logical_and(sr_mask, df.loc[:, 'AUX_label1D'].eq(options_dict['SIGNAL_LABEL']).to_numpy())
    signal_sr_scores = df.loc[signal_sr_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    signal_sr_weights = df.loc[signal_sr_mask, 'AUX_eventWeight'].to_numpy()
    # Bkg events inside SR
    bkg_sr_mask = np.isfinite(np.logical_and(sr_mask, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL'])))
    bkg_sr_scores = df.loc[bkg_sr_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    bkg_sr_weights = df.loc[bkg_sr_mask, 'AUX_eventWeight'].to_numpy()
    # Bkg events outside SR
    bkg_sideband_mask = np.isfinite(np.logical_and(sideband_mask, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL'])))
    bkg_sideband_scores = df.loc[bkg_sideband_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    bkg_sideband_weights = df.loc[bkg_sideband_mask, 'AUX_eventWeight'].to_numpy()

    startstops = copy.deepcopy(options_dict['STARTSTOPS'])
    for zoom in range(options_dict['N_ZOOM']):
        steps = [
            np.linspace(
                startstops[i][0], startstops[i][1], options_dict['N_STEPS'], endpoint=True
            ) for i in range(len(options_dict['TRANSFORM_COLUMNS']))
        ]
        cuts = np.array(ak.to_list(ak.cartesian(steps, axis=0)))
        foms = np.zeros(np.shape(cuts)[0])
        all_cuts.extend(cuts)

        foms = brute_force(
            signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
            bkg_sideband_scores, bkg_sideband_weights, 
            cuts, foms
        )
        all_foms.extend(foms)

        index = np.argmax(foms)
        fom, cut = foms[index], cuts[index]
        step_sizes = [(stop - start) / options_dict['N_STEPS'] for start, stop in startstops]
        startstops = [[cut[i] - step_sizes[i], cut[i] + step_sizes[i]] for i in range(len(step_sizes))]

        if fom > best_fom: best_fom = fom; best_cut = cut

    return best_fom, best_cut





# def compute_cuts1D(df: pd.DataFrame, cat_mask: pd.DataFrame, options_dict: dict):
#     pass_fom = np.logical_and(cat_mask, fom_mask(df))
#     pass_sideband = np.logical_and(cat_mask, sideband_nonres_mask(df))

#     best_fom, best_cut = 0., 0.

#     start1, stop1 = options_dict['START1'], options_dict['STOP1']
#     for zoom in range(options_dict['N_ZOOM']):
#         foms, cuts = [], []

#         for cut1 in np.linspace(start1, stop1, options_dict['N_STEPS'], endpoint=True):
#             cuts.append(cut1)

#             pass_cut = np.logical_and(pass_fom, df.loc[:, options_dict['TRANSFORM_COLUMNS'][0]].gt(cut1))
#             sideband_cut = np.logical_and(pass_sideband, df.loc[:, options_dict['TRANSFORM_COLUMNS'][0]].gt(cut1))

#             signal = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].eq(options_dict['SIGNAL_LABEL'])), 'AUX_eventWeight'].sum()
#             bkg = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL'])), 'AUX_eventWeight'].sum()
#             sideband_nonres = df.loc[sideband_cut, 'AUX_eventWeight'].sum()
#             fom = fom_s_over_sqrt_b(signal, bkg) if sideband_nonres > 8. else 0.
#             foms.append(fom)

#         index = np.argmax(foms)
#         fom, cut = foms[index], cuts[index]
#         step_size1 = (stop1 - start1) / options_dict['N_STEPS']
#         start1, stop1 = cut - step_size1, cut + step_size1

#         if fom > best_fom: best_fom = fom; best_cut = cut

#     return best_fom, (best_cut)

# def compute_cuts2D(df: pd.DataFrame, cat_mask: pd.DataFrame, options_dict: dict):
#     pass_fom = np.logical_and(cat_mask, fom_mask(df))
#     pass_sideband = np.logical_and(cat_mask, sideband_nonres_mask(df))

#     best_fom, best_cut = 0., (0., 0.)

#     start1, stop1 = options_dict['START1'], options_dict['STOP1']
#     start2, stop2 = options_dict['START2'], options_dict['STOP2']
#     for zoom in range(options_dict['N_ZOOM']):
#         foms, cuts = [], []
        
#         for cut1 in np.linspace(start1, stop1, options_dict['N_STEPS'], endpoint=True):

#             pass_cut = np.logical_and(pass_fom, df.loc[:, options_dict['TRANSFORM_COLUMNS'][0]].gt(cut1))
#             sideband_cut = np.logical_and(pass_sideband, df.loc[:, options_dict['TRANSFORM_COLUMNS'][0]].gt(cut1))
            
#             _foms_, _cuts_ = [], []
#             for cut2 in np.linspace(start2, stop2, options_dict['N_STEPS'], endpoint=True):
#                 _cuts_.append( (cut1, cut2) )

#                 pass_cut = np.logical_and(pass_cut, df.loc[:, options_dict['TRANSFORM_COLUMNS'][1]].gt(cut2))
#                 sideband_cut = np.logical_and(pass_sideband, df.loc[:, options_dict['TRANSFORM_COLUMNS'][1]].gt(cut2))

#                 signal = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].eq(options_dict['SIGNAL_LABEL'])), 'AUX_eventWeight'].sum()
#                 bkg = df.loc[np.logical_and(pass_cut, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL'])), 'AUX_eventWeight'].sum()
#                 sideband_nonres = df.loc[sideband_cut, 'AUX_eventWeight'].sum()
#                 fom = fom_s_over_sqrt_b(signal, bkg) if (sideband_nonres > 8. or zoom != options_dict['N_ZOOM']-1) and np.isfinite(fom_s_over_sqrt_b(signal, bkg)) else 0.
#                 # print(f"{_cuts_[-1]}: fom = {fom}; signal = {signal}, bkg = {bkg}, nonres sideband = {sideband_nonres}")
#                 _foms_.append(fom)
#             foms.append(_foms_)
#             cuts.append(_cuts_)
        
#         index = np.unravel_index(np.argmax(foms), np.shape(foms))
#         fom, cut = foms[index[0]][index[1]], cuts[index[0]][index[1]]
#         step_size1, step_size2 = (stop1 - start1) / options_dict['N_STEPS'], (stop2 - start2) / options_dict['N_STEPS']
#         start1, stop1 = cut[0] - step_size1, cut[0] + step_size1
#         start2, stop2 = cut[1] - step_size2, cut[1] + step_size2

#         if fom > best_fom: best_fom = fom; best_cut = cut

#     print(f"{best_fom:.2f}, ({best_cut[0]:.4f}, {best_cut[1]:.4f})")
#     return best_fom, best_cut
