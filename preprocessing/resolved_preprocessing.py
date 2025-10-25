import copy

import awkward as ak
import numpy as np
import vector as vec
vec.register_awkward()

from preprocessing_utils import (
    deltaPhi, deltaEta, 
    match_sample
)

################################


resolved_bTagWPs = {
    '2022*preEE': ("btagPNetB", {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961}),
    '2022*postEE': ("btagPNetB", {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664}),
    '2023*preBPix': ("btagPNetB", {'L': 0.0358, 'M': 0.1917, 'T': 0.6172, 'XT': 0.7515, 'XXT': 0.9659}),
    '2023*postBPix': ("btagPNetB", {'L': 0.0359, 'M': 0.1919, 'T': 0.6133, 'XT': 0.7544, 'XXT': 0.9688}),
    # '2024': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739})
    '2024': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739, 'XXXT': 0.9983, 'XXXXT': 0.9987, 'XXXXXT': 0.9989})  # XMT calculated to have ggF HH kl-1p00 lead *AND* sublead bjets pass with 25% efficiency, XXXT was calculated to have ggF HH kl-1p00 lead *OR* sublead bjets pass with 25% efficiency, XXMT calculated for 35% efficiency, XXXXT calculated for 10% efficiency, XXXXXT calculated for 5% efficiency
}

FILL_VALUE = -999
NUM_JETS = 10
# PREFACTORS = ['nonRes', 'nonResReg', 'nonResReg_DNNpair']
PREFACTORS = ['nonRes', 'nonResReg_DNNpair']  # 'nonResReg'

################################


def fix_UParT_field(sample, prefactor='nonRes'):
    for bjet_type in ['lead', 'sublead']:
        if f"{prefactor}{bjet_type}_bjet_btagUParTAK4B" in sample.fields:
            sample[f"{prefactor}_{bjet_type}_bjet_btagUParTAK4B"] = sample[f"{prefactor}{bjet_type}_bjet_btagUParTAK4B"]

# Variables to add for resolved training
def add_bTagWP_resolved(sample, era, prefactor='nonRes'):
    bTagVar, WP_dict = resolved_bTagWPs[match_sample(era, resolved_bTagWPs.keys())]
    
    for bjet_type in ['lead', 'sublead']:
        for WPname, WP in WP_dict.items():
            sample[f"{prefactor}_{bjet_type}_bjet_bTagWP{WPname}"] = ak.where(sample[f"{prefactor}_{bjet_type}_bjet_{bTagVar}"] > WP, 1, 0)

def jet_mask(sample, i, prefactor='nonRes'):
    return (
        ak.where(
            sample[f'{prefactor}_lead_bjet_jet_idx'] != i, True, False
        ) & ak.where(
            sample[f'{prefactor}_sublead_bjet_jet_idx'] != i, True, False
        ) & ak.where(sample[f'jet{i}_mass'] != FILL_VALUE, True, False)
    )

def zh_isr_jet(sample, dijet_4mom, jet_4moms, prefactor='nonRes'):
    min_total_pt = ak.Array([FILL_VALUE for _ in range(ak.num(sample['event'], axis=0))])
    isr_jet_4mom = copy.deepcopy(jet_4moms['jet1_4mom'])

    for i in range(1, NUM_JETS+1):
        jet_i_mask = jet_mask(sample, i, prefactor=prefactor)

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
    return isr_jet_4mom, (min_total_pt != FILL_VALUE)

def max_nonbjet_btag(sample, prefactor='nonRes'):
    max_btag_score = ak.Array([0. for _ in range(ak.num(sample['event'], axis=0))])

    for i in range(1, NUM_JETS+1):
        jet_i_mask = jet_mask(sample, i, prefactor=prefactor)

        larger_btag_bool = jet_i_mask & (
            sample[f'jet{i}_btagPNetB'] > max_btag_score
        )

        max_btag_score = ak.where(
            larger_btag_bool, sample[f'jet{i}_btagPNetB'], max_btag_score
        )
    return max_btag_score

def add_vars_resolved(sample, filepath):

    # Regressed jet kinematics #
    jet_4moms = {}
    for i in range(1, NUM_JETS+1):
        jet_4moms[f'jet{i}_4mom'] = ak.zip(
            {
                'rho': sample[f'jet{i}_pt'],
                'phi': sample[f'jet{i}_phi'],
                'eta': sample[f'jet{i}_eta'],
                'tau': sample[f'jet{i}_mass'],
            }, with_name='Momentum4D'
        )

    for prefactor in PREFACTORS:

        good_bjets = ak.zip({
            'lead': (sample[f'{prefactor}_lead_bjet_pt'] != FILL_VALUE),
            'sublead': (sample[f'{prefactor}_sublead_bjet_pt'] != FILL_VALUE),
            'dijet': (sample[f'{prefactor}_lead_bjet_pt'] != FILL_VALUE) & (sample[f'{prefactor}_sublead_bjet_pt'] != FILL_VALUE)
        })
    
        # Regressed bjet kinematics #
        bjet_4moms = {}
        for field in ['lead', 'sublead']:
            bjet_4moms[f'{field}_bjet_4mom'] = ak.zip(
                {
                    'rho': sample[f'{prefactor}_{field}_bjet_pt'], # rho is synonym for pt
                    'phi': sample[f'{prefactor}_{field}_bjet_phi'],
                    'eta': sample[f'{prefactor}_{field}_bjet_eta'],
                    'tau': sample[f'{prefactor}_{field}_bjet_mass'], # tau is synonym for mass
                }, with_name='Momentum4D'
            )

        # Regressed dijet kinematics #
        dijet_4mom = bjet_4moms['lead_bjet_4mom'] + bjet_4moms['sublead_bjet_4mom']

        # Nonres BDT variables #
        for field in ['lead', 'sublead']:
            # photon variables
            sample[f'{field}_sigmaE_over_E'] = sample[f'{field}_energyErr'] / (sample[f'{field}_pt'] * np.cosh(sample[f'{field}_eta']))
            
            # bjet variables
            sample[f'{prefactor}_{field}_bjet_pt_over_Mjj'] = ak.where(
                good_bjets[field], 
                sample[f'{prefactor}_{field}_bjet_pt'] / sample[f'{prefactor}_dijet_mass'],
                FILL_VALUE
            )
            sample[f'{prefactor}_{field}_bjet_sigmapT_over_pT'] = ak.where(
                good_bjets[field],
                sample[f'{prefactor}_{field}_bjet_PNetRegPtRawRes'] / sample[f'{prefactor}_{field}_bjet_pt'],
                FILL_VALUE
            )


        # mHH variables #
        sample[f'{prefactor}_pt_balance'] = ak.where(
            good_bjets['dijet'],
            sample[f'{prefactor}_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample[f'{prefactor}_lead_bjet_pt'] + sample[f'{prefactor}_sublead_bjet_pt']),
            FILL_VALUE
        )


        # VH variables #
        sample[f'{prefactor}_DeltaPhi_jj'] = ak.where(
            good_bjets['dijet'],
            deltaPhi(sample[f'{prefactor}_lead_bjet_phi'], sample[f'{prefactor}_sublead_bjet_phi']),
            FILL_VALUE
        )
        sample[f'{prefactor}_DeltaEta_jj'] = ak.where(
            good_bjets['dijet'],
            deltaEta(sample[f'{prefactor}_lead_bjet_eta'], sample[f'{prefactor}_sublead_bjet_eta']),
            FILL_VALUE
        )


        # ISR-like jet variables
        isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample, dijet_4mom, jet_4moms)
        sample[f'{prefactor}_isr_jet_pt'] = ak.where(  # pt of isr jet
            isr_jet_bool & good_bjets['dijet'], 
            isr_jet_4mom.pt, 
            FILL_VALUE
        )
        sample[f'{prefactor}_DeltaPhi_isr_jet_z'] = ak.where(  # phi angle between isr jet and z candidate
            isr_jet_bool & good_bjets['dijet'],
            deltaPhi(isr_jet_4mom.phi, sample[f'{prefactor}_dijet_phi']), 
            FILL_VALUE
        )


        # max non-bjet btag score -> sets lower limit for resampling #
        sample[f'{prefactor}_max_nonbjet_btag'] = max_nonbjet_btag(sample, prefactor=prefactor)

        fix_UParT_field(sample, prefactor=prefactor)
        add_bTagWP_resolved(sample, filepath, prefactor=prefactor)

        sample['pass_mva-0.7'] = ak.where(
            (sample['lead_mvaID'] > -0.7)
            & (sample['sublead_mvaID'] > -0.7),
            1, 0
        )
        sample[f'{prefactor}_resolved_BDT_mask'] = ak.where(
            sample[f'{prefactor}_has_two_btagged_jets'] & (sample['pass_mva-0.7'] > 0)
            & (
                sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
            ) & (
                (
                    (sample[f'{prefactor}_lead_bjet_bTagWPT'] > 0)
                    & (sample[f'{prefactor}_sublead_bjet_bTagWPT'] > 0)
                ) if match_sample(filepath, {'W*HTo2G', 'ZH*To2G'}) is not None else (sample['mass'] > 0)
            ) & (
                (
                    sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90']
                    | sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95']
                ) if 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90' in sample.fields else (sample['mass'] > 0)
            ),
            1, 0
        )

        del bjet_4moms, dijet_4mom
