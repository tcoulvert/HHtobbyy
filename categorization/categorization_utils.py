# Stdlib packages
import copy

# Common Py packages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# HEP packages
import awkward as ak

################################


SR_CUTS = [122.5, 127.]
SIDEBAND_CUTS = [[100., 120.], [130., 200.]]

################################


def fom_mask(df: pd.DataFrame):
    return np.logical_and(df.loc[:, 'AUX_mass'].ge(SR_CUTS[0]).to_numpy(), df.loc[:, 'AUX_mass'].le(SR_CUTS[1]).to_numpy())

def sideband_nonres_mask(df: pd.DataFrame):
    mass_cut = np.logical_or(
        np.logical_and(
            df.loc[:, 'AUX_mass'].gt(SIDEBAND_CUTS[0][0]).to_numpy(), df.loc[:, 'AUX_mass'].lt(SIDEBAND_CUTS[0][1]).to_numpy()
        ),
        np.logical_and(
            df.loc[:, 'AUX_mass'].gt(SIDEBAND_CUTS[1][0]).to_numpy(), df.loc[:, 'AUX_mass'].lt(SIDEBAND_CUTS[1][1]).to_numpy()
        )
    )
    if np.any(df.loc[:, 'AUX_sample_name'].eq('Data').to_numpy()):
        sample_cut = df.loc[:, 'AUX_sample_name'].eq('Data').to_numpy()
    else:
        sample_cut = np.logical_or(
            df.loc[:, 'AUX_sample_name'].eq('TTGG').to_numpy(),  # TTGG
            np.logical_or(
                np.logical_or(df.loc[:, 'AUX_sample_name'].eq('GJet').to_numpy(), df.loc[:, 'AUX_sample_name'].eq('GGJets').to_numpy()),  # GGJets or GJet
                df.loc[:, 'AUX_sample_name'].eq('DDQCDGJets').to_numpy()  # DDQCD GJet or GGJets
            )
        )
    return np.logical_and(mass_cut, sample_cut)


def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)

def fom_s_over_b(s, b):
    return s / b


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


def exp_func(x, a, b):
    return a * np.exp(b * x)

def brute_force(
    signal_sr_scores: np.ndarray, signal_sr_weights: np.ndarray, 
    bkg_sr_scores: np.ndarray, bkg_sr_weights: np.ndarray, 
    bkg_sideband_scores: np.ndarray, bkg_sideband_weights: np.ndarray, bkg_sideband_mass: np.ndarray,
    cuts: np.ndarray, foms: np.ndarray, cutdir: np.ndarray, sideband_fit: bool=False
):
    lt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '<']) if any(_cutdir_ == '<' for _cutdir_ in cutdir) else None
    gt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '>']) if any(_cutdir_ == '>' for _cutdir_ in cutdir) else None

    signal_lt_scores, bkg_lt_scores, sideband_lt_scores = lt_scores(signal_sr_scores), lt_scores(bkg_sr_scores), lt_scores(bkg_sideband_scores)
    signal_gt_scores, bkg_gt_scores, sideband_gt_scores = gt_scores(signal_sr_scores), gt_scores(bkg_sr_scores), gt_scores(bkg_sideband_scores)
    lt_cuts, gt_cuts = lt_scores(cuts), gt_scores(cuts)

    ncuts, ndims = cuts.shape
    jump_to_cut = 0
    best_dim_foms, best_dim_cuts = np.array([-1.]*ndims), np.array([cuts[0]]*ndims)


    def apply_cuts(_lt_scores_, _gt_scores_):
        if any(_cutdir_ == '<' for _cutdir_ in cutdir) and any(_cutdir_ == '>' for _cutdir_ in cutdir):
            pass_cut_bool = np.logical_and(np.all(_lt_scores_ < lt_cuts[i:i+1], axis=1), np.all(_gt_scores_ > gt_cuts[i:i+1], axis=1))
        elif any(_cutdir_ == '<' for _cutdir_ in cutdir):
            pass_cut_bool = np.all(_lt_scores_ < lt_cuts[i:i+1], axis=1)
        elif any(_cutdir_ == '>' for _cutdir_ in cutdir):
            pass_cut_bool = np.all(_gt_scores_ > gt_cuts[i:i+1], axis=1)
        else: raise NotImplementedError(f"Provided cut directions for discriminator can only be \'<\' or \'>\', your cut directions are {cutdir}")
        return pass_cut_bool


    for i in range(ncuts):
        if i < jump_to_cut: continue

        bkg_sideband_bool = apply_cuts(sideband_lt_scores, sideband_gt_scores)

        if (
            np.sum(bkg_sideband_weights[bkg_sideband_bool]) > 10. 
            # and (
            #     np.sum(bkg_sideband_weights[bkg_sideband_bool][bkg_sideband_mass[bkg_sideband_bool] < SIDEBAND_CUTS[0][1]])
            #     / np.sum(bkg_sideband_weights[bkg_sideband_bool][bkg_sideband_mass[bkg_sideband_bool] > SIDEBAND_CUTS[1][0]])
            # ) > 0.25
        ):
            signal_sr_bool = apply_cuts(signal_lt_scores, signal_gt_scores)
            bkg_sr_bool = apply_cuts(bkg_lt_scores, bkg_gt_scores)

            if sideband_fit:
                binwidth = 5
                _hist_, _bins_ = np.histogram(bkg_sideband_mass[bkg_sideband_bool], bins=np.arange(100., 180., binwidth), weights=bkg_sideband_weights[bkg_sideband_bool])
                p0 = (_hist_[0], -0.1)
                params, _ = curve_fit(exp_func, _bins_[:-1]-_bins_[0], _hist_, p0=p0, sigma=np.where(np.isfinite(_hist_**-1), _hist_**-1, 0.76))
                y_pred = exp_func(SR_CUTS-_bins_[0], a=params[0], b=params[1])
                est_yield = (SR_CUTS[1] - SR_CUTS[0]) * (y_pred[0] - 0.5*(y_pred[0] - y_pred[1]))
                ascii_hist(bkg_sideband_mass[bkg_sideband_bool], bins=np.arange(100., 180., binwidth), weights=bkg_sideband_weights[bkg_sideband_bool], fit=exp_func(_bins_[:-1]-_bins_[0], a=params[0], b=params[1]))
                print(f"y = {params[0]:.2f}e^({params[1]:.2f}x)")
                print(f"  -> est. yield of non-res in SR = {est_yield}")
            else: est_yield = 0.

            foms[i] = fom_s_over_b(
                np.sum(signal_sr_weights[signal_sr_bool]), np.sum(bkg_sr_weights[bkg_sr_bool])+est_yield,
            )
        else: foms[i] = 0.

        if i > 0 and (foms[i-1] > foms[i] or (foms[i-1] == foms[i] and foms[i-1] != 0.)):
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
    bkg_sideband_mass = df.loc[bkg_sideband_mask, 'AUX_mass'].to_numpy()

    startstops = copy.deepcopy(options_dict['STARTSTOPS'])
    for zoom in range(options_dict['N_ZOOM']):
        print(f"Zoom {zoom}")
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
            bkg_sideband_scores, bkg_sideband_weights, bkg_sideband_mass,
            cuts, foms, np.array(cutdir), sideband_fit=np.any(df.loc[:, 'AUX_sample_name'].eq('Data').to_numpy())
        )
        all_cuts.extend(cuts[foms >= 0.])
        all_foms.extend(foms[foms >= 0.])

        step_sizes = [(stop - start) / options_dict['N_STEPS'] for start, stop in startstops]
        startstops = [[cut[i] - step_sizes[i], cut[i] + step_sizes[i]] for i in range(len(step_sizes))]

        if fom > best_fom: best_fom = fom; best_cut = cut; print(f"best cut = {cut}, best fom = {fom}")

    return best_fom.item(), best_cut.tolist()
