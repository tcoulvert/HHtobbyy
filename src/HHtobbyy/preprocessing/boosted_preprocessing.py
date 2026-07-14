# Stdlib packages
import re

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

# Workspace packages
from HHtobbyy.preprocessing.preprocessing_utils import deltaPhi, deltaEta
from HHtobbyy.workspace_utils.retrieval_utils import FILL_VALUE, match_sample

################################


boosted_bbTagWPs = {
    '202[23]': ("particleNet_XbbVsQCD", {'L': 0.4, 'M': 0.83, 'T': 0.89, 'XT': 0.925, 'XXT': 0.96}),
    '(201[x678])|(202[45])': ("globalParT3_XbbVsQCD", {'L': 0.57, 'M': 0.8, 'T': 0.86, 'XT': 0.91, 'XXT': 0.96}),
}
boosted_fjmass_corr = {
    '202[23]': "particleNet_massCorr",
    '(201[x678])|(202[45])': "globalParT3_massCorrX2p"
}

NUM_FATJETS = 4
SELECTION = ['pt > 300', 'tau21 < 0.75', 'msoftdrop > 30', 'particleNet_XbbVsQCD > 0.4', 'globalParT3_XbbVsQCD > 0.6', 'genMatched_Hbb > 0', 'eta <= 2.4']

################################


# Variables to add for boosted training
def add_bbTagNanov15_boosted(df, era):
    bbTagVar, _ = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    if bbTagVar == 'globalParT3_XbbVsQCD':
        for i in range(1, NUM_FATJETS+1):
            df[f'fatjet{i}_globalParT3_XbbVsQCD'] = df[f'fatjet{i}_globalParT3_Xbb'] / (df[f'fatjet{i}_globalParT3_Xbb'] + df[f'fatjet{i}_globalParT3_QCD'])

def add_bbTagWP_boosted(df, era, prefactor):
    bbTagVar, WP_dict = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    for i, (WPname, WP) in enumerate(WP_dict.items()):
        df[f"{prefactor}_fatjet_selected_bbTagWP{WPname}"] = np.where(df[f"{prefactor}_fatjet_selected_{bbTagVar}"] > WP, 1, 0)
        if i == 0: df[f"{prefactor}_fatjet_selected_bbTagWP"] = ak.zeros_like(df["pt"])
        df[f"{prefactor}_fatjet_selected_bbTagWP"] = np.where(df[f"{prefactor}_fatjet_selected_{bbTagVar}"] > WP, i+1, df[f"{prefactor}_fatjet_selected_bbTagWP"])

def select_fatjets(df, era):
    fatjet_fields = [col[col.find('fatjet1_')+len('fatjet1_'):] for col in df.columns if re.match('fatjet1', col) is not None]
    fatjets = ak.zip({
        fatjet_field: np.concatenate([df[f'fatjet{i}_{fatjet_field}'][:, np.newaxis] for i in range(1, NUM_FATJETS+1)], axis=1)
        for fatjet_field in fatjet_fields
    })

    bbTagVar, _ = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]

    selection_mask = ak.ones_like(fatjets[fatjets.columns[0]])
    for cut in SELECTION:
        var, direction, value = cut.split(' ')
        if var == 'genMatched_Hbb':
            if var not in df.columns or ak.sum(fatjets[var] > 0) == 0: continue
        elif var == 'particleNet_XbbVsQCD':
            if any(['globalParT3_XbbVsQCD' in col for col in df.columns]): continue
        elif var == 'globalParT3_XbbVsQCD':
            if not any(['globalParT3_XbbVsQCD' in col for col in df.columns]): continue
        if direction == '<':
            selection_mask = np.logical_and(selection_mask, fatjets[var] < float(value))
        elif direction == '<=':
            selection_mask = np.logical_and(selection_mask, fatjets[var] <= float(value))
        elif direction == '==':
            selection_mask = np.logical_and(selection_mask, fatjets[var] == float(value))
        elif direction == '>=':
            selection_mask = np.logical_and(selection_mask, fatjets[var] >= float(value))
        elif direction == '>':
            selection_mask = np.logical_and(selection_mask, fatjets[var] > float(value))
        else: raise NotImplementedError(f"The direction you passed is unknown: {direction}. Use \'<\', \'<=\', \'==\', \'>=\', or \'>\'.")
    selection_fatjets = fatjets[selection_mask]
    selection_fatjets = selection_fatjets[ak.argsort(selection_fatjets[bbTagVar])]

    selected_fatjets = ak.firsts(selection_fatjets)
    return selected_fatjets[~ak.is_none(selected_fatjets)], ~ak.is_none(selected_fatjets)


def add_n_fatjets_final(df):
    df["n_fatjets_final"] = np.zeros(len(df))
    for i in range(1, NUM_FATJETS+1):
        eta_cut = (np.abs(df[f'fatjet{i}_eta']) <= 2.4)
        df["n_fatjets_final"] = np.where(
            eta_cut, df["n_fatjets_final"]+1, df["n_fatjets_final"]
        )

def add_vars_boosted(df: pd.DataFrame, filepath: str, prefactor: str, **kwargs):
    # Fatjet tau ratio and Xbb vs QCD discriminator #
    for i in range(1, NUM_FATJETS+1):
        df[f'fatjet{i}_tau21'] = df[f'fatjet{i}_tau2'] / df[f'fatjet{i}_tau1']
        df[f'fatjet{i}_tau32'] = df[f'fatjet{i}_tau3'] / df[f'fatjet{i}_tau2']
        if f'fatjet{i}_mass_raw' in df.columns:
            df[f'fatjet{i}_corrmass'] = df[f'fatjet{i}_mass_raw'] * df[f'fatjet{i}_{boosted_fjmass_corr[match_sample(filepath, boosted_fjmass_corr.keys())]}']

    add_bbTagNanov15_boosted(df, filepath)

    selected_fatjets, good_fatjets = select_fatjets(df, filepath)
    df = df.loc[good_fatjets].reset_index(drop=True)
    for col in selected_fatjets.fields: df[f'fatjet_selected_{col}'] = selected_fatjets[col]

    # / USE REG MASS IF AVAILABLE OTHERWISE SOFTDROP /
    if f'fatjet_selected_globalParT3_massCorrX2p' in df.columns and f'fatjet_selected_mass_raw' in df.columns:
        df[f'fatjet_selected_mass_regressed'] = np.where(
            np.isnan(df[f'fatjet_selected_globalParT3_massCorrX2p']),
            df[f'fatjet_selected_msoftdrop'],
            df[f'fatjet_selected_corrmass']
        )
    else: 
        df[f'fatjet_selected_mass_regressed'] = df[f'fatjet_selected_msoftdrop']
    df[f'fatjet_selected_mass_regressed'] = df[f'fatjet_selected_mass_regressed']

    # (Di)Photon - fatjet angular variables #
    for photon_type, photon_field_prefix in [('gg', ''), ('g1', 'lead_'), ('g2', 'sublead_')]:
        df[f'deltaEta_{photon_type}_fj'] = deltaEta(df[f'{photon_field_prefix}eta'], df[f'fatjet_selected_eta'])
        df[f'deltaPhi_{photon_type}_fj'] = deltaPhi(df[f'{photon_field_prefix}phi'], df[f'fatjet_selected_phi'])
        df[f'deltaR_{photon_type}_fj'] = ( df[f'deltaEta_{photon_type}_fj']**2 + df[f'deltaPhi_{photon_type}_fj']**2 )**0.5

    for subj_type, subj_field in [('subj1', 'subjet1'), ('subj2', 'subjet2')]:
        df[f'deltaEta_{subj_type}_gg'] = deltaEta(df[f'fatjet_selected_{subj_field}_eta'], df[f'{photon_field_prefix}eta'])
        df[f'deltaPhi_{subj_type}_gg'] = deltaPhi(df[f'fatjet_selected_{subj_field}_phi'], df[f'{photon_field_prefix}phi'])
        df[f'deltaR_{subj_type}_gg'] = ( df[f'deltaEta_{subj_type}_gg']**2 + df[f'deltaPhi_{subj_type}_gg']**2 )**0.5

    df[f'deltaEta_subj1_subj2'] = deltaEta(df[f'fatjet_selected_subjet1_eta'], df[f'fatjet_selected_subjet2_eta'])
    df[f'deltaPhi_subj1_subj2'] = deltaPhi(df[f'fatjet_selected_subjet1_phi'], df[f'fatjet_selected_subjet2_phi'])
    df[f'deltaR_subj1_subj2'] = ( df[f'deltaEta_subj1_subj2']**2 + df[f'deltaPhi_subj1_subj2']**2 )**0.5
    df['deltaEta_g1_g2'] = deltaEta(df['lead_eta'], df['sublead_eta'])

    # Fatjet bb WP variable #
    add_bbTagWP_boosted(df, filepath, prefactor)

    # n_fatjets_final
    add_n_fatjets_final(df)

    # Mask for training #
    pass_presel_mask = np.logical_and(
        np.logical_and(
            np.logical_and(df['lead_mvaID'] > -0.7, df['sublead_mvaID'] > -0.7),
            np.logical_and(df["mass"] > 100, df["mass"] < 180)
        ),
        df['n_fatjets_final'] > 0
    )
    df = df.loc[pass_presel_mask].reset_index(drop=True)
    return df

