# Stdlib packages
import copy
import time

# Common Py packages
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


def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)

def fom_s_over_b(s, b):
    return s / b


def brute_force(
    signal_sr_scores, signal_sr_weights, 
    bkg_sr_scores, bkg_sr_weights, 
    bkg_sideband_scores, bkg_sideband_weights, 
    cuts, foms, cutdir
):
    lt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '<']) if any(_cutdir_ == '<' for _cutdir_ in cutdir) else None
    gt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '>']) if any(_cutdir_ == '>' for _cutdir_ in cutdir) else None

    signal_lt_scores, bkg_lt_scores, sideband_lt_scores = lt_scores(signal_sr_scores), lt_scores(bkg_sr_scores), lt_scores(bkg_sideband_scores)
    signal_gt_scores, bkg_gt_scores, sideband_gt_scores = gt_scores(signal_sr_scores), gt_scores(bkg_sr_scores), gt_scores(bkg_sideband_scores)
    lt_cuts, gt_cuts = lt_scores(cuts), gt_scores(cuts)

    ncuts, ndims = cuts.shape
    jump_to_cut = 0
    best_dim_foms, best_dim_cuts = np.array([-1.]*ndims), np.array([cuts[0]]*ndims)
    for i in range(ncuts):
        if i < jump_to_cut: continue

        if any(_cutdir_ == '<' for _cutdir_ in cutdir) and any(_cutdir_ == '>' for _cutdir_ in cutdir):
            signal_sr_bool = np.logical_and(np.all(signal_lt_scores < lt_cuts[i:i+1], axis=1), np.all(signal_gt_scores > gt_cuts[i:i+1], axis=1))
            bkg_sr_bool = np.logical_and(np.all(bkg_lt_scores < lt_cuts[i:i+1], axis=1), np.all(bkg_gt_scores > gt_cuts[i:i+1], axis=1))
            bkg_sideband_bool = np.logical_and(np.all(sideband_lt_scores < lt_cuts[i:i+1], axis=1), np.all(sideband_gt_scores > gt_cuts[i:i+1], axis=1))
        elif any(_cutdir_ == '<' for _cutdir_ in cutdir):
            signal_sr_bool = np.all(signal_lt_scores < lt_cuts[i:i+1], axis=1)
            bkg_sr_bool = np.all(bkg_lt_scores < lt_cuts[i:i+1], axis=1)
            bkg_sideband_bool = np.all(sideband_lt_scores < lt_cuts[i:i+1], axis=1)
        elif any(_cutdir_ == '>' for _cutdir_ in cutdir):
            signal_sr_bool = np.all(signal_gt_scores > gt_cuts[i:i+1], axis=1)
            bkg_sr_bool = np.all(bkg_gt_scores > gt_cuts[i:i+1], axis=1)
            bkg_sideband_bool = np.all(sideband_gt_scores > gt_cuts[i:i+1], axis=1)
        else: raise NotImplementedError(f"Provided cut directions for discriminator can only be \'<\' or \'>\', your cut directions are {cutdir}")

        foms[i] = fom_s_over_b(
            np.sum(signal_sr_weights[signal_sr_bool]), np.sum(bkg_sr_weights[bkg_sr_bool]),
        ) if np.sum(bkg_sideband_weights[bkg_sideband_bool]) > 8. else 0.

        if i > 0 and (foms[i-1] > foms[i] or foms[i-1] == foms[i]):
            for j in range(ndims+1):
                if j == ndims: return best_dim_foms[j-1], best_dim_cuts[j-1]
                if j == 0:
                    if foms[i-1] > best_dim_foms[j]:
                        best_dim_foms[j] = foms[i-1]; best_dim_cuts[j] = cuts[i-1]
                        jump_index = -2; break
                elif best_dim_foms[j-1] > best_dim_foms[j]: 
                    best_dim_foms[j] = best_dim_foms[j-1]; best_dim_cuts[j] = best_dim_cuts[j-1]
                    best_dim_foms[j-1] = -1.; jump_index = -(j+1); break

            jump_to_cut = i + np.argmax(cuts[i:, jump_index] != cuts[i, jump_index])
    return best_dim_foms[np.argmax(best_dim_foms)], best_dim_cuts[np.argmax(best_dim_foms)]

def grid_search(df: pd.DataFrame, cat_mask: np.ndarray, options_dict: dict, cutdir: list, prev_cuts: list=None):

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
        steps = [
            np.linspace(
                startstops[i][1], startstops[i][0], options_dict['N_STEPS'], endpoint=False
            )[::-1] for i in range(len(options_dict['TRANSFORM_COLUMNS']))
        ]
        cuts = np.array(ak.to_list(ak.cartesian(steps, axis=0)))
        if prev_cuts is not None:
            cuts = np.array([cut for cut in cuts if all((cut[i] < prev_cut[i] if cutdir[i] == '>' else cut[i] > prev_cut[i]) for prev_cut in prev_cuts for i in range(len(prev_cut)))])
        foms = -np.ones(np.shape(cuts)[0])

        fom, cut = brute_force(
            signal_sr_scores, signal_sr_weights, 
            bkg_sr_scores, bkg_sr_weights, 
            bkg_sideband_scores, bkg_sideband_weights, 
            cuts, foms, cutdir  # nb.typed.List(cutdir)
        )
        all_cuts.extend(cuts[foms >= 0.])
        all_foms.extend(foms[foms >= 0.])

        step_sizes = [(stop - start) / options_dict['N_STEPS'] for start, stop in startstops]
        startstops = [[cut[i] - step_sizes[i], cut[i] + step_sizes[i]] for i in range(len(step_sizes))]

        if fom > best_fom: best_fom = fom; best_cut = cut; print(f"best cut = {cut}, best fom = {fom}")

    return best_fom.item(), best_cut.tolist()
