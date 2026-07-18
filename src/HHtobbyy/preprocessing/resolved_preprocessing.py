# Stdlib packages
import copy

# Common Py packages
import numpy as np  
import pandas as pd

# HEP packages
import awkward as ak
import vector as vec
vec.register_awkward()

# Workspace packages
from HHtobbyy.preprocessing.preprocessing_utils import deltaPhi, deltaEta
from HHtobbyy.workspace_utils.retrieval_utils import FILL_VALUE, match_sample

################################

resolved_bTagWPs = {
    # MC
    '2016*preVFP': ("btagUParTAK4B", {'L': 0.0387, 'M': 0.1847, 'T': 0.5467, 'XT': 0.6777, 'XXT': 0.9218}),
    '2016*postVFP': ("btagUParTAK4B", {'L': 0.0400, 'M': 0.1898, 'T': 0.5538, 'XT': 0.6872, 'XXT': 0.9353}),
    '2017': ("btagUParTAK4B", {'L': 0.0331, 'M': 0.1776, 'T': 0.5755, 'XT': 0.7274, 'XXT': 0.9666}),
    '2018': ("btagUParTAK4B", {'L': 0.0308, 'M': 0.1610, 'T': 0.5405, 'XT': 0.6992, 'XXT': 0.9655}),

    '2022*(preEE)|((Data|Era)[CD])': ("btagPNetB", {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961, '3XT': 0.9986, '4XT': 0.999}),
    '2022*(postEE)|((Data|Era)[EFG])': ("btagPNetB", {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664, '3XT': 0.9986, '4XT': 0.999}),
    '2023*(preBPix)|((Data|Era)[C])': ("btagPNetB", {'L': 0.0358, 'M': 0.1917, 'T': 0.6172, 'XT': 0.7515, 'XXT': 0.9659, '3XT': 0.9986, '4XT': 0.999}),
    '2023*(postBPix)|((Data|Era)[D])': ("btagPNetB", {'L': 0.0359, 'M': 0.1919, 'T': 0.6133, 'XT': 0.7544, 'XXT': 0.9688, '3XT': 0.9986, '4XT': 0.999}),
    '2024': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739, '3XT': 0.9983, '4XT': 0.9987}),  # 3XT was calculated to have ggF HH kl-1p00 lead *OR* sublead bjets pass with 25% efficiency, 4XT calculated for 10% efficiency
    '2025': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739, '3XT': 0.9983, '4XT': 0.9987}), 
}

NUM_JETS = 10

################################


# Variables to add for resolved training
def add_bTagWP_resolved(df: pd.DataFrame, filepath: str, prefactor: str):
    bTagVar, WP_dict = resolved_bTagWPs[match_sample(filepath, resolved_bTagWPs.keys())]
    
    for bjet_type in ['lead', 'sublead']:
        for i, (WPname, WP) in enumerate(WP_dict.items()):
            df[f"{prefactor}_{bjet_type}_bjet_bTagWP{WPname}"] = np.where(df[f"{prefactor}_{bjet_type}_bjet_{bTagVar}"] > WP, 1, 0)
            if WPname not in ['L', 'M', 'T', 'XT', 'XXT']: continue
            elif i == 0: df[f"{prefactor}_{bjet_type}_bjet_bTagWP"] = np.zeros(len(df))
            df[f"{prefactor}_{bjet_type}_bjet_bTagWP"] = np.where(df[f"{prefactor}_{bjet_type}_bjet_{bTagVar}"] > WP, i+1, df[f"{prefactor}_{bjet_type}_bjet_bTagWP"])


def jet_mask(df, i, prefactor: str):
    return (
        ak.where(
            df[f'{prefactor}_lead_bjet_jet_idx'].to_numpy() != i, True, False
        ) & ak.where(
            df[f'{prefactor}_sublead_bjet_jet_idx'].to_numpy() != i, True, False
        ) & ak.where(df[f'jet{i}_mass'].to_numpy() != FILL_VALUE, True, False)
    )

def zh_isr_jet(df, dijet_4mom, jet_4moms, prefactor: str):
    min_total_pt = FILL_VALUE * ak.ones_like(dijet_4mom.pt)
    isr_jet_4mom = copy.deepcopy(jet_4moms['jet1_4mom'])

    for i in range(1, NUM_JETS+1):
        jet_i_mask = jet_mask(df, i, prefactor=prefactor)

        z_jet_4mom = dijet_4mom + jet_4moms[f'jet{i}_4mom']

        better_isr_bool = (
            (z_jet_4mom.pt < min_total_pt) | (min_total_pt == FILL_VALUE)
        ) & jet_i_mask
        min_total_pt = ak.where(
            better_isr_bool, z_jet_4mom.pt, min_total_pt
        )
        isr_jet_4mom = ak.where(
            better_isr_bool, jet_4moms[f'jet{i}_4mom'], isr_jet_4mom
        )
    return isr_jet_4mom, np.asarray(ak.to_numpy(min_total_pt != FILL_VALUE, allow_missing=False), dtype=bool)

def add_vars_resolvedMLP(df: pd.DataFrame, filepath: str, prefactor: str='', **kwargs):
    add_bTagWP_resolved(df, filepath, prefactor=prefactor)

    ### BEGIN Manos variables ###
    df[f"{prefactor}_diphoton_PtOverM_ggjj"] = df["pt"] / df[f"{prefactor}_HHbbggCandidate_mass"]
    df[f"{prefactor}_dijet_PtOverM_ggjj"] = df[f"{prefactor}_dijet_pt"] / df[f"{prefactor}_HHbbggCandidate_mass"]

    df[f"{prefactor}_lead_bjet_over_M_regressed"] = df[f"{prefactor}_lead_bjet_pt"] / df[f"{prefactor}_dijet_mass_DNNreg"]
    df[f"{prefactor}_sublead_bjet_over_M_regressed"] = df[f"{prefactor}_sublead_bjet_pt"] / df[f"{prefactor}_dijet_mass_DNNreg"]
    ### END Manos variables ###

    # Mask for training #
    pass_presel_mask = np.logical_and(
        np.logical_and(df['lead_mvaID'] > -0.7, df['sublead_mvaID'] > -0.7),
        np.logical_and(df[f"{prefactor}_dijet_mass_DNNreg"] > 80, df[f"{prefactor}_dijet_mass_DNNreg"] < 190)
    )
    df = df.loc[pass_presel_mask].reset_index(drop=True)
    return df

def add_vars_resolvedBDT(df: pd.DataFrame, filepath: str, prefactor: str='', **kwargs):
    add_bTagWP_resolved(df, filepath, prefactor=prefactor)

    # Nonres BDT variables #
    for field in ['lead', 'sublead']:
        # photon variables
        df[f'{field}_sigmaE_over_E'] = df[f'{field}_energyErr'] / (df[f'{field}_pt'] * np.cosh(df[f'{field}_eta']))
        
        # bjet variables
        df[f'{prefactor}_{field}_bjet_pt_over_Mjj'] = df[f'{prefactor}_{field}_bjet_pt'] / df[f'{prefactor}_dijet_mass_DNNreg']
        df[f'{prefactor}_{field}_bjet_sigmapT_over_pT'] = df[f'{prefactor}_{field}_bjet_PNetRegPtRawRes'] / df[f'{prefactor}_{field}_bjet_pt']


    # mHH variables #
    df[f'{prefactor}_pt_balance'] = df[f'{prefactor}_HHbbggCandidate_pt'] / (df['lead_pt'] + df['sublead_pt'] + df[f'{prefactor}_lead_bjet_pt'] + df[f'{prefactor}_sublead_bjet_pt'])


    # VH variables #
    df[f'{prefactor}_DeltaPhi_jj'] = deltaPhi(df[f'{prefactor}_lead_bjet_phi'], df[f'{prefactor}_sublead_bjet_phi'])
    df[f'{prefactor}_DeltaEta_jj'] = deltaEta(df[f'{prefactor}_lead_bjet_eta'], df[f'{prefactor}_sublead_bjet_eta'])


    # Regressed jet kinematics #
    jet_4moms = {}
    for i in range(1, NUM_JETS+1):
        jet_4moms[f'jet{i}_4mom'] = ak.zip(
            {
                'rho': df[f'jet{i}_pt'].to_numpy(),
                'phi': df[f'jet{i}_phi'].to_numpy(),
                'eta': df[f'jet{i}_eta'].to_numpy(),
                'tau': df[f'jet{i}_mass'].to_numpy(),
            }, with_name='Momentum4D'
        )
    # Regressed bjet kinematics #
    bjet_4moms = {}
    for field in ['lead', 'sublead']:
        bjet_4moms[f'{field}_bjet_4mom'] = ak.zip(
            {
                'rho': df[f'{prefactor}_{field}_bjet_pt'].to_numpy(), # rho is synonym for pt
                'phi': df[f'{prefactor}_{field}_bjet_phi'].to_numpy(),
                'eta': df[f'{prefactor}_{field}_bjet_eta'].to_numpy(),
                'tau': df[f'{prefactor}_{field}_bjet_mass'].to_numpy(), # tau is synonym for mass
            }, with_name='Momentum4D'
        )
    # Regressed dijet kinematics #
    dijet_4mom = bjet_4moms['lead_bjet_4mom'] + bjet_4moms['sublead_bjet_4mom']

    # ISR-like jet variables
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(df, dijet_4mom, jet_4moms, prefactor=prefactor)
    df[f'{prefactor}_isr_jet_pt'] = np.where(isr_jet_bool, ak.to_numpy(isr_jet_4mom.pt, allow_missing=False), FILL_VALUE)
    df[f'{prefactor}_DeltaPhi_isr_jet_z'] = np.where(isr_jet_bool, deltaPhi(ak.to_numpy(isr_jet_4mom.phi, allow_missing=False), ak.to_numpy(dijet_4mom.phi, allow_missing=False)), FILL_VALUE)


    # Mask for training #
    pass_presel_mask = np.logical_and(
        np.logical_and(df['lead_mvaID'] > -0.7, df['sublead_mvaID'] > -0.7),
        np.logical_and(df[f"{prefactor}_dijet_mass_DNNreg"] > 80, df[f"{prefactor}_dijet_mass_DNNreg"] < 190)
    )
    df = df.loc[pass_presel_mask].reset_index(drop=True)
    return df

def add_vars_resolvedBDTLbTag(df: pd.DataFrame, filepath: str, prefactor: str='', **kwargs):
    df = add_vars_resolvedBDT(df, filepath, prefactor, **kwargs)
    
    # Mask for training #
    pass_presel_mask = (df[f'{prefactor}_lead_bjet_bTagWP'] > 0)
    df = df.loc[pass_presel_mask].reset_index(drop=True)
    return df