# Stdlib packages
import copy

# Common Py packages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad

# HEP packages
import awkward as ak
import hist

################################


def mass_cut(df: pd.DataFrame, cuts: list[float], aux_prefix: str) -> np.ndarray[bool]:
    return np.logical_and(
        df.loc[:, f'{aux_prefix}mass'].ge(cuts[0]).to_numpy(), 
        df.loc[:, f'{aux_prefix}mass'].le(cuts[1]).to_numpy()
    )

#############################################################
# Figure of Merit
def fom_s_over_sqrt_b(s, b):
    return s / np.sqrt(b)
def fom_s_over_b(s, b):
    return s / b
def fom_zscore(s, b):
    return np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s))

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
# Sideband fit for nonRes bkg estimaton
def est_yield(mass: np.ndarray, weight: np.ndarray, sr_masscut: list[float], sb_masscut: list[float]):
    """Return (b_sr, var_b_sr) using linear interpolation from SB densities.

    Weights each sideband density by its centroid distance from the SR centroid,
    matching the estimate one would obtain from a linear fit to the sidebands.
    """
    m = mass
    w = weight

    L = (m >= sb_masscut[0]) & (m < sr_masscut[0])
    R = (m >= sr_masscut[1]) & (m < sb_masscut[1])

    sumw_L = w[L].sum()
    sumw2_L = (w[L] ** 2).sum()
    sumw_R = w[R].sum()
    sumw2_R = (w[R] ** 2).sum()

    width_L = max(1e-9, sr_masscut[0] - sb_masscut[0])
    width_R = max(1e-9, sb_masscut[1] - sr_masscut[1])
    width_SR = max(1e-9, sr_masscut[1] - sr_masscut[0])

    dens_L = sumw_L / width_L if width_L > 0 else 0.0
    var_dens_L = sumw2_L / (width_L ** 2) if width_L > 0 else 0.0

    dens_R = sumw_R / width_R if width_R > 0 else 0.0
    var_dens_R = sumw2_R / (width_R ** 2) if width_R > 0 else 0.0

    have_L = sumw_L > 0.0
    have_R = sumw_R > 0.0

    if have_L and have_R:
      centroid_L = 0.5 * (sb_masscut[0] + sr_masscut[0])
      centroid_R = 0.5 * (sr_masscut[1] + sb_masscut[1])
      centroid_SR = 0.5 * (sr_masscut[0] + sr_masscut[1])
      span = max(1e-9, centroid_R - centroid_L)
      w_L = (centroid_R - centroid_SR) / span
      w_R = (centroid_SR - centroid_L) / span
      dens = w_L * dens_L + w_R * dens_R
      var_dens = w_L ** 2 * var_dens_L + w_R ** 2 * var_dens_R
    elif have_L:
      dens = dens_L
      var_dens = var_dens_L
    elif have_R:
      dens = dens_R
      var_dens = var_dens_R
    else:
      return 0.0

    b_sr = dens * width_SR
    var_b_sr = var_dens * (width_SR ** 2)

    return float(b_sr)

def brute_force(
    # Input NPs
    signal_sr_scores: np.ndarray, signal_sr_weights: np.ndarray, 
    res_sr_scores: np.ndarray, res_sr_weights: np.ndarray, 
    nonres_sb_scores: np.ndarray, nonres_sb_weights: np.ndarray, nonres_sb_mass: np.ndarray,
    # Cut options
    cutdir: np.ndarray, SR_masscut: list[float], SB_masscut: list[float],
    # Optimization options
    fom, min_nonres_sideband: float,
    # Output arrays
    cuts: np.ndarray, foms: np.ndarray,
):
    lt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '<']) if any(_cutdir_ == '<' for _cutdir_ in cutdir) else None
    gt_scores = lambda scores: np.column_stack([scores[:, j] for j in range(scores.shape[1]) if cutdir[j] == '>']) if any(_cutdir_ == '>' for _cutdir_ in cutdir) else None

    signal_lt_scores, res_lt_scores, nonres_lt_scores = lt_scores(signal_sr_scores), lt_scores(res_sr_scores), lt_scores(nonres_sb_scores)
    signal_gt_scores, res_gt_scores, nonres_gt_scores = gt_scores(signal_sr_scores), gt_scores(res_sr_scores), gt_scores(nonres_sb_scores)
    print(cuts)
    print(cuts.shape)
    lt_cuts, gt_cuts = lt_scores(cuts), gt_scores(cuts)


    def apply_cuts(_lt_scores_, _gt_scores_, i):
        if any(_cutdir_ == '<' for _cutdir_ in cutdir) and any(_cutdir_ == '>' for _cutdir_ in cutdir):
            pass_cut_bool = np.logical_and(np.all(_lt_scores_ < lt_cuts[i:i+1], axis=1), np.all(_gt_scores_ > gt_cuts[i:i+1], axis=1))
        elif any(_cutdir_ == '<' for _cutdir_ in cutdir):
            pass_cut_bool = np.all(_lt_scores_ < lt_cuts[i:i+1], axis=1)
        elif any(_cutdir_ == '>' for _cutdir_ in cutdir):
            pass_cut_bool = np.all(_gt_scores_ > gt_cuts[i:i+1], axis=1)
        else: raise NotImplementedError(f"Provided cut directions for discriminator can only be \'<\' or \'>\', your cut directions are {cutdir}")
        return pass_cut_bool


    for i in range(cuts.shape[0]):
        nonres_sb_bool = apply_cuts(nonres_lt_scores, nonres_gt_scores, i)
        # print(f"Cut = {cuts[i]}; Num SB = {np.sum(nonres_sb_weights[nonres_sb_bool]):.2f}")

        if np.sum(nonres_sb_weights[nonres_sb_bool]) > min_nonres_sideband:
            signal_sr_bool = apply_cuts(signal_lt_scores, signal_gt_scores, i)
            res_sr_bool = apply_cuts(res_lt_scores, res_gt_scores, i)

            sb_est_yield = est_yield(nonres_sb_mass[nonres_sb_bool], nonres_sb_weights[nonres_sb_bool], SR_masscut, SB_masscut)

            foms[i] = fom(np.sum(signal_sr_weights[signal_sr_bool]), np.sum(res_sr_weights[res_sr_bool]) + sb_est_yield)
        else: foms[i] = -1.

    return foms[np.argmax(foms)], cuts[np.argmax(foms)]

def grid_search(MCsignal: pd.DataFrame, MCres: pd.DataFrame, MCnonRes: pd.DataFrame, catconfig, prev_cuts: list[float]=None):

    best_fom, best_cut, best_iteration = np.float64(-1.), np.zeros(len(catconfig.transform_names)), -1

    all_foms, all_cuts = [], []

    # Signal events inside SR
    signal_sr_mask = mass_cut(MCsignal, catconfig.SR_masscut, catconfig.dfdataset.aux_var_prefix)
    signal_sr_scores = MCsignal.loc[signal_sr_mask, catconfig.transform_names].to_numpy()
    if len(signal_sr_scores.shape) == 1: signal_sr_scores = signal_sr_scores[:, np.newaxis]
    signal_sr_weights = MCsignal.loc[signal_sr_mask, f'{catconfig.dfdataset.aux_var_prefix}eventWeight'].to_numpy()
    # Res events inside SR
    res_sr_mask = mass_cut(MCres, catconfig.SR_masscut, catconfig.dfdataset.aux_var_prefix)
    res_sr_scores = MCres.loc[res_sr_mask, catconfig.transform_names].to_numpy()
    if len(res_sr_scores.shape) == 1: res_sr_scores = res_sr_scores[:, np.newaxis]
    res_sr_weights = MCres.loc[res_sr_mask, f'{catconfig.dfdataset.aux_var_prefix}eventWeight'].to_numpy()
    # nonRes events in SB
    nonres_sb_mask = np.logical_and(
        mass_cut(MCnonRes, catconfig.SB_masscut, catconfig.dfdataset.aux_var_prefix),
        ~mass_cut(MCnonRes, catconfig.SR_masscut, catconfig.dfdataset.aux_var_prefix)
    )
    nonres_sb_scores = MCnonRes.loc[nonres_sb_mask, catconfig.transform_names].to_numpy()
    if len(nonres_sb_scores.shape) == 1: nonres_sb_scores = nonres_sb_scores[:, np.newaxis]
    nonres_sb_weights = MCnonRes.loc[nonres_sb_mask, f'{catconfig.dfdataset.aux_var_prefix}eventWeight'].to_numpy()
    nonres_sb_mass = MCnonRes.loc[nonres_sb_mask, f'{catconfig.dfdataset.aux_var_prefix}mass'].to_numpy()


    max_iterations = int((catconfig.n_dims // catconfig.method_options['step_size']) + 1)
    for iteration in range(1, max_iterations):
        print(f"Iteration {iteration}")

        if catconfig.n_dims > 1: 
            Nm1D = catconfig.n_dims - 1
            Nm1D_arrs = [np.arange(0, iteration+1)] * (Nm1D)
            Nm1D_combinations = np.stack(np.meshgrid(*Nm1D_arrs), axis=-1).reshape(-1, Nm1D)
            Nm1D_combinations = Nm1D_combinations[np.logical_and(Nm1D_combinations.sum(axis=1) < iteration, np.all(Nm1D_combinations != 0, axis=1))]
            ND_combinations = np.hstack((Nm1D_combinations, (iteration - Nm1D_combinations.sum(axis=1))[:, np.newaxis]))
        else:
            ND_combinations = np.array([[iteration]])
        
        cuts = catconfig.method_options['step_size'] * ND_combinations
        for iD, cutdir in enumerate(catconfig.cutdir):
            if cutdir == '>': cuts[:, iD] = 1. - cuts[:, iD]
        if prev_cuts is not None:
            cuts = np.array([cut for cut in cuts if all((cut[i] < prev_cut[i] if catconfig.cutdir[i] == '>' else cut[i] > prev_cut[i]) for prev_cut in prev_cuts for i in range(len(prev_cut)))])
        print(cuts)
        if len(cuts) == 0: continue

        foms = -np.ones(np.shape(cuts)[0])

        fom, cut = brute_force(
            # Input NPs
            signal_sr_scores, signal_sr_weights, 
            res_sr_scores, res_sr_weights, 
            nonres_sb_scores, nonres_sb_weights, nonres_sb_mass,
            # Cut options
            np.array(catconfig.cutdir), catconfig.SR_masscut, catconfig.SB_masscut,
            # Optimization options
            catconfig.get_fom(), catconfig.min_nonres_sideband,
            # Output arrays
            cuts, foms
        )
        all_cuts.extend(cuts[foms >= 0.])
        all_foms.extend(foms[foms >= 0.])

        if fom - best_fom > 0.002: best_fom = fom; best_cut = cut; best_iteration = iteration; print(f"best cut = {cut}, best fom = {fom}")
        elif best_iteration > 0 and iteration - best_iteration >= catconfig.method_options['patience']: break

    return best_fom, best_cut
