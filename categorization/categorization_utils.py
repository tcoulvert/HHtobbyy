# Stdlib packages
import copy

# Common Py packages
import numba as nb
import numba_cuda as cuda
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


@nb.njit()
def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)

@nb.njit()
def fom_s_over_b(s, b):
    return s / b


@nb.njit(parallel=True)
def brute_force(
    signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
    bkg_sideband_scores, bkg_sideband_weights, 
    cuts, foms, cutdir
):
    signal_yields, bkg_yields, sideband_yields = np.zeros_like(foms), np.zeros_like(foms), np.zeros_like(foms)
    for i in nb.prange(cuts.shape[0]):
        # Signal yield in SR
        for j in nb.prange(signal_sr_scores.shape[0]):
            pass_cut_bool = True
            for k in nb.prange(signal_sr_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
                ( (signal_sr_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
                | ( (signal_sr_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
            )
            signal_yields[i] += pass_cut_bool * signal_sr_weights[j]
        # Background yield in SR
        for j in nb.prange(bkg_sr_scores.shape[0]):
            pass_cut_bool = True
            for k in nb.prange(bkg_sr_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
                ( (bkg_sr_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
                | ( (bkg_sr_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
            )
            bkg_yields[i] += pass_cut_bool * bkg_sr_weights[j]
        # Background yield outside SR
        for j in nb.prange(bkg_sideband_scores.shape[0]):
            pass_cut_bool = True
            for k in nb.prange(bkg_sideband_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
                ( (bkg_sideband_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
                | ( (bkg_sideband_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
            )
            sideband_yields[i] += pass_cut_bool * bkg_sideband_weights[j]
    for i in nb.prange(foms.shape[0]):
        foms[i] = fom_s_over_b(signal_yields[i], bkg_yields[i]) if sideband_yields[i] > 8. else 0.
    return foms

# @cuda.jit
# def brute_force_cuda(
#     signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
#     bkg_sideband_scores, bkg_sideband_weights, 
#     cuts, foms, cutdir
# ):
#     signal_yields, bkg_yields, sideband_yields = np.zeros_like(foms), np.zeros_like(foms), np.zeros_like(foms)
#     for i in nb.prange(cuts.shape[0]):
#         # Signal yield in SR
#         for j in nb.prange(signal_sr_scores.shape[0]):
#             pass_cut_bool = True
#             for k in nb.prange(signal_sr_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
#                 ( (signal_sr_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
#                 | ( (signal_sr_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
#             )
#             signal_yields[i] += pass_cut_bool * signal_sr_weights[j]
#         # Background yield in SR
#         for j in nb.prange(bkg_sr_scores.shape[0]):
#             pass_cut_bool = True
#             for k in nb.prange(bkg_sr_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
#                 ( (bkg_sr_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
#                 | ( (bkg_sr_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
#             )
#             bkg_yields[i] += pass_cut_bool * bkg_sr_weights[j]
#         # Background yield outside SR
#         for j in nb.prange(bkg_sideband_scores.shape[0]):
#             pass_cut_bool = True
#             for k in nb.prange(bkg_sideband_scores.shape[1]): pass_cut_bool = pass_cut_bool & ( 
#                 ( (bkg_sideband_scores[j][k] > cuts[i][k]) & (cutdir[k] == '>') )
#                 | ( (bkg_sideband_scores[j][k] < cuts[i][k]) & (cutdir[k] == '<') )
#             )
#             sideband_yields[i] += pass_cut_bool * bkg_sideband_weights[j]
#     for i in nb.prange(foms.shape[0]):
#         foms[i] = fom_s_over_b(signal_yields[i], bkg_yields[i]) if sideband_yields[i] > 8. else 0.
#     return foms

def brute_force_python(
    signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
    bkg_sideband_scores, bkg_sideband_weights, 
    cuts, foms, cutdir
):
    lt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '<'])
    gt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '>'])

    signal_lt_scores, bkg_lt_scores, sideband_lt_scores = lt_scores(signal_sr_scores), lt_scores(bkg_sr_scores), lt_scores(bkg_sideband_scores)
    signal_gt_scores, bkg_gt_scores, sideband_gt_scores = gt_scores(signal_sr_scores), gt_scores(bkg_sr_scores), gt_scores(bkg_sideband_scores)
    lt_cuts, gt_cuts = lt_scores(cuts), gt_scores(cuts)

    jump_to_cut = -1
    best_fom, best_cut = 0., cuts[0]
    for i in range(cuts.shape[0]):
        if i < jump_to_cut: continue

        foms[i] = fom_s_over_b(
            np.sum(signal_sr_weights[np.logical_and(signal_lt_scores < lt_cuts, signal_gt_scores > gt_cuts)]),
            np.sum(bkg_sr_weights[np.logical_and(bkg_lt_scores < lt_cuts, bkg_gt_scores > gt_cuts)]),
        ) if np.sum(bkg_sideband_weights[np.logical_and(sideband_lt_scores < lt_cuts, sideband_gt_scores > gt_cuts)]) > 8. else 0.

        if i > 0 and foms[i-1] > foms[i]: 
            if foms[i-1] < best_fom: 
                if cuts.shape[1] < 3: break
                jump_to_cut = np.argmax(cuts[:, cuts.shape[1]-3] != cuts[i, cuts.shape[1]-3])
            else: 
                if cuts.shape[1] < 2: break
                jump_to_cut = np.argmax(cuts[:, cuts.shape[1]-2] != cuts[i, cuts.shape[1]-2])
        else:
            best_fom, best_cut = foms[i], cuts[i]

    return best_fom, best_cut


def grid_search(df: pd.DataFrame, cat_mask: np.ndarray, options_dict: dict, cutdir: list):
    sr_mask = np.logical_and(cat_mask, fom_mask(df))
    sideband_mask = np.logical_and(cat_mask, sideband_nonres_mask(df))
    assert not np.any(np.logical_and(sr_mask, sideband_mask)), print(f"Overlap between SR and Sideband definitions... THIS IS VERY BAD")

    best_fom, best_cut = 0., [0. for _ in options_dict['TRANSFORM_COLUMNS']]

    all_foms, all_cuts = [], []

    # Signal events inside SR
    signal_sr_mask = np.logical_and(sr_mask, df.loc[:, 'AUX_label1D'].eq(options_dict['SIGNAL_LABEL']).to_numpy())
    signal_sr_scores = df.loc[signal_sr_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    signal_sr_weights = df.loc[signal_sr_mask, 'AUX_eventWeight'].to_numpy()
    # Bkg events inside SR
    bkg_sr_mask = np.logical_and(sr_mask, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL']).to_numpy())
    bkg_sr_scores = df.loc[bkg_sr_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    bkg_sr_weights = df.loc[bkg_sr_mask, 'AUX_eventWeight'].to_numpy()
    # Bkg events outside SR
    bkg_sideband_mask = np.logical_and(sideband_mask, df.loc[:, 'AUX_label1D'].ne(options_dict['SIGNAL_LABEL']).to_numpy())
    bkg_sideband_scores = df.loc[bkg_sideband_mask, options_dict['TRANSFORM_COLUMNS']].to_numpy()
    bkg_sideband_weights = df.loc[bkg_sideband_mask, 'AUX_eventWeight'].to_numpy()

    startstops = copy.deepcopy(options_dict['STARTSTOPS'])
    for zoom in range(options_dict['N_ZOOM']):
        print(f"Zoom {zoom}")
        steps = [
            np.linspace(
                startstops[i][0], startstops[i][1], options_dict['N_STEPS'], endpoint=True
            ) for i in range(len(options_dict['TRANSFORM_COLUMNS']))
        ]
        cuts = np.array(ak.to_list(ak.cartesian(steps, axis=0)))
        foms = np.zeros(np.shape(cuts)[0])
        # all_cuts.extend(cuts)

        # if cuda:
        #     threadsperblock = 256
        #     blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
        #     brute_force_cuda[blockspergrid, threadsperblock](
        #         signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
        #         bkg_sideband_scores, bkg_sideband_weights, 
        #         cuts, foms, nb.typed.List(cutdir)
        #     )
        # else:
        #     brute_force(
        #         signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
        #         bkg_sideband_scores, bkg_sideband_weights, 
        #         cuts, foms, nb.typed.List(cutdir)
        #     )
        # all_foms.extend(foms)
        cut, fom = brute_force_python(
            signal_sr_scores, signal_sr_weights, bkg_sr_scores, bkg_sr_weights, 
            bkg_sideband_scores, bkg_sideband_weights, 
            cuts, foms, nb.typed.List(cutdir)
        )

        # index = np.argmax(foms)
        # fom, cut = foms[index], cuts[index]
        step_sizes = [(stop - start) / options_dict['N_STEPS'] for start, stop in startstops]
        startstops = [[cut[i] - step_sizes[i], cut[i] + step_sizes[i]] for i in range(len(step_sizes))]

        if fom > best_fom: best_fom = fom; best_cut = cut; print(f"best cut = {cut}, best fom = {fom}")

    return best_fom.item(), best_cut.tolist()
