# Stdlib packages
import copy
import re

# Common Py packages
import numpy as np

# HEP packages
import awkward as ak
import vector as vec

################################


from preprocessing_utils import (
    deltaPhi, deltaEta, 
    match_sample
)

################################


vec.register_awkward()

boosted_bbTagWPs = {
    '202[2-3]': ("particleNet_XbbVsQCD", {'L': 0.4, 'M': 0.6, 'T': 0.8, 'XT': 0.9, 'XXT': 0.95}),
    '201[6-8]|202[4-5]': ("globalParT3_XbbVsQCD", {'L': 0.25, 'M': 0.5, 'T': 0.8, 'XT': 0.9, 'XXT': 0.95}),
}
boosted_fjmass_corr = {
    '202[2-3]': "particleNet_massCorr",
    '201[6-8]|202[4-5]': "globalParT3_massCorrGeneric"
}

FILL_VALUE = -999
NUM_FATJETS = 4
PREFACTORS = ['Res', 'Res_DNNpair']

################################


# Variables to add for boosted training
def add_bbTagNanov15_boosted(sample, era):
    bbTagVar, _ = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    if bbTagVar == 'globalParT3_XbbVsQCD':
        for i in range(1, NUM_FATJETS+1):
            sample[f'fatjet{i}_globalParT3_XbbVsQCD'] = sample[f'fatjet{i}_globalParT3_Xbb'] / (sample[f'fatjet{i}_globalParT3_Xbb'] + sample[f'fatjet{i}_globalParT3_QCD'])

def add_bbTagWP_boosted(sample, era):
    bbTagVar, WP_dict = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    for i, (WPname, WP) in enumerate(WP_dict.items()):
        if i == 0: sample["fatjet_selected_bbTagWP"] = ak.zeros_like(sample["pt"])
        sample["fatjet_selected_bbTagWP"] = ak.where(sample[f"fatjet_selected_{bbTagVar}"] > WP, i, sample["fatjet_selected_bbTagWP"])

def select_fatjets(sample, era):
    fatjet_fields = [field[field.find('fatjet1_')+len('fatjet1_'):] for field in sample.fields if re.match('fatjet1', field) is not None]
    fatjets = ak.zip({
        fatjet_field: ak.concatenate([sample[f'fatjet{i}_{fatjet_field}'][:, np.newaxis] for i in range(1, NUM_FATJETS+1)], axis=1)
        for fatjet_field in fatjet_fields
    })

    bbTagVar, _ = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]

    selection_mask = (fatjets['pt'] > 250) & (fatjets['tau21'] < 0.6)
    selection_fatjets = fatjets[selection_mask]
    selection_fatjets = selection_fatjets[ak.argsort(selection_fatjets[bbTagVar])]

    selected_fatjets = ak.firsts(selection_fatjets)
    return selected_fatjets, ~ak.is_none(selected_fatjets)
    
def fatjet_mask(sample, i):
    fatjet_selected_4mom = ak.zip(
        {
            'rho': sample['fatjet_selected_pt'],
            'phi': sample['fatjet_selected_phi'],
            'eta': sample['fatjet_selected_eta'],
            'tau': sample['fatjet_selected_mass'],
        }, with_name='Momentum4D'
    )
    fatjet_i_4mom = ak.zip(
        {
            'rho': sample[f'fatjet{i}_pt'],
            'phi': sample[f'fatjet{i}_phi'],
            'eta': sample[f'fatjet{i}_eta'],
            'tau': sample[f'fatjet{i}_mass'],
        }, with_name='Momentum4D'
    )
    return fatjet_selected_4mom.deltaR(fatjet_i_4mom) > 0.01

def max_nonselectedfatjet_bbTag(sample, era):
    bbTagVar, _ = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    max_bbTag_score = ak.Array([0. for _ in range(ak.num(sample['event'], axis=0))])

    for i in range(1, NUM_FATJETS+1):
        fatjet_i_mask = fatjet_mask(sample, i)

        larger_bbTag_bool = fatjet_i_mask & (
            sample[f'fatjet{i}_{bbTagVar}'] > max_bbTag_score
        )

        max_bbTag_score = ak.where(
            larger_bbTag_bool, sample[f'fatjet{i}_{bbTagVar}'], max_bbTag_score
        )
    return max_bbTag_score

def add_vars_boosted(sample, filepath):
    prefactors = [prefactor for prefactor in PREFACTORS if any(match_sample(field, PREFACTORS) == prefactor for field in sample.fields)]

    # Fatjet tau ratio and Xbb vs QCD discriminator #
    for i in range(1, NUM_FATJETS+1):
        sample[f'fatjet{i}_tau21'] = sample[f'fatjet{i}_tau2'] / sample[f'fatjet{i}_tau1']
        sample[f'fatjet{i}_tau32'] = sample[f'fatjet{i}_tau3'] / sample[f'fatjet{i}_tau2']
        sample[f'fatjet{i}_corrmass'] = sample[f'fatjet{i}_mass'] * sample[f'fatjet{i}_{boosted_fjmass_corr[match_sample(filepath, boosted_fjmass_corr.keys())]}']

    add_bbTagNanov15_boosted(sample, filepath)

    selected_fatjets, good_fatjets = select_fatjets(sample, filepath)
    for field in selected_fatjets:
        sample[f'fatjet_selected_{field}'] = ak.where(good_fatjets, selected_fatjets[field], FILL_VALUE)

    # (Di)Photon - fatjet angular variables #
    for photon_type, photon_field_prefix in [('gg', ''), ('g1', 'lead_'), ('g2', 'sublead_')]:
        sample[f'deltaEta_{photon_type}_fj'] = ak.where(
            good_fatjets,
            deltaEta(sample[f'{photon_field_prefix}eta'], sample['fatjet_selected_eta']),
            FILL_VALUE
        )
        sample[f'deltaPhi_{photon_type}_fj'] = ak.where(
            good_fatjets,
            deltaPhi(sample[f'{photon_field_prefix}phi'], sample['fatjet_selected_phi']),
            FILL_VALUE
        )
        sample[f'deltaR_{photon_type}_fj'] = ak.where(
            good_fatjets,
            ( sample[f'deltaEta_{photon_type}_fj']**2 + sample[f'deltaPhi_{photon_type}_fj']**2 )**0.5,
            FILL_VALUE
        )

    # Pt ratios #
    sample['pt_Over_FatjetPt'] = sample['pt'] / sample['fatjet_selected_pt']
    for prefactor in prefactors:
        sample[f'{prefactor}_fatjet_pt_balance'] = ak.where(
            good_fatjets,
            sample[f'{prefactor}_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['fatjet_selected_pt']),
            FILL_VALUE
        )

    # Fatjet bb WP variable #
    add_bbTagWP_boosted(sample, filepath)

    # Max non-bb fatjet bbTag score -> sets lower limit for resampling #
    sample['max_nonselectedfatjet_bbtag'] = max_nonselectedfatjet_bbTag(sample, filepath)

    # Mask for training #
    sample['pass_mva-0.7'] = ak.where(
        (sample['lead_mvaID'] > -0.7)
        & (sample['sublead_mvaID'] > -0.7),
        1, 0
    )
    sample[f'boosted_BDT_mask'] = ak.where(
        good_fatjets & (sample['pass_mva-0.7'] > 0)
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        ) & (
            (
                sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90']
                | sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95']
            ) if 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90' in sample.fields else (sample['mass'] > 0)
        ),
        1, 0
    )
