import copy
import glob
import json
import math
import os
import re

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector as vec
vec.register_awkward()

# lpc_redirector = "root://cmseos.fnal.gov/"
# lxplus_redirector = "root://eosuser.cern.ch/"
# lxplus_fileprefix = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v2"
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v3.1/"
FILL_VALUE = -999
NUM_JETS = 10
FORCE_RERUN = True

DATASETTYPE = {
    'Resolved', 'Boosted'
}

def add_vars(sample, datasettype):
    if datasettype == 'Resolved':
        return add_vars_resolved(sample)
    elif datasettype == 'Boosted':
        return add_vars_boosted(sample)
    else:
        raise NotImplementedError(f"Datasettype you requested ({datasettype}) is not implemented. We only have implemented: {DATASETTYPE.keys()}")

def add_vars_boosted(sample):
    sample['simple_boosted_PNmass'] = (
        sample['Res_has_atleast_one_fatjet']
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        ) & (  # fatjet cuts
            (sample['fatjet1_pt'] > 250)
            & (
                (sample['fatjet1_mass'] > 100)
                & (sample['fatjet1_mass'] < 160)
            ) & (sample['fatjet1_particleNet_XbbVsQCD'] > 0.8)
        ) & (  # good photon cuts (for boosted regime)
            (sample['lead_mvaID'] > 0.)
            & (sample['sublead_mvaID'] > 0.)
        )
    )

    sample['simple_boosted_SDmass'] = (
        sample['Res_has_atleast_one_fatjet']
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        ) & (  # fatjet cuts
            (sample['fatjet1_pt'] > 250)
            & (
                (sample['fatjet1_msoftdrop'] > 100)
                & (sample['fatjet1_msoftdrop'] < 160)
            ) & (sample['fatjet1_particleNet_XbbVsQCD'] > 0.8)
        ) & (  # good photon cuts (for boosted regime)
            (sample['lead_mvaID'] > 0.)
            & (sample['sublead_mvaID'] > 0.)
        )
    )

def add_vars_resolved(sample):
    
    def ak_sign(ak_array, inverse=False):
        if not inverse:
            return ak.where(ak_array < 0, -1, 1)
        else:
            return ak.where(ak_array < 0, 1, -1)
            
    def ak_abs(ak_array):
        valid_entry_mask = ak.where(ak_array != FILL_VALUE, True, False)
        abs_ak_array = ak.where(ak_array > 0, ak_array, -ak_array)
        return ak.where(valid_entry_mask, abs_ak_array, FILL_VALUE)
        
    def deltaPhi(phi1, phi2):
        # angle1 and angle2 are (-pi, pi]
        # Convention: clockwise is (+), anti-clockwise is (-)
        subtract_angles = phi1 - phi2
        return ak.where(ak_abs(subtract_angles) <= math.pi, subtract_angles, subtract_angles + 2*math.pi*ak_sign(subtract_angles, inverse=True))
    
    def deltaEta(eta1, eta2):
        return ak_abs(eta1 - eta2)
    
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
                ak.where(z_jet_4mom.pt < min_total_pt, True, False)
                | ak.where(min_total_pt == FILL_VALUE, True, False)
            ) & jet_i_mask
            min_total_pt = ak.where(
                better_isr_bool, z_jet_4mom.pt, min_total_pt
            )
            isr_jet_4mom = ak.where(
                better_isr_bool, jet_4moms[f'jet{i}_4mom'], isr_jet_4mom
            )
        return isr_jet_4mom, ak.where(min_total_pt != FILL_VALUE, True, False)
    
    def robustParT(sample, bjet_type='lead'):
        robust_parT = ak.Array([FILL_VALUE for _ in range(ak.num(sample['event'], axis=0))])

        for i in range(1, NUM_JETS+1):
            robust_parT = ak.where(sample[f"nonRes_{bjet_type}_bjet_jet_idx"] == i, sample[f"jet{i}_btagRobustParTAK4B"], robust_parT)

        return robust_parT
    
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
    
    # Regressed bjet kinematics #
    bjet_4moms = {}
    for field in ['lead', 'sublead']:
        bjet_4moms[f'{field}_bjet_4mom'] = ak.zip(
            {
                'rho': sample[f'nonRes_{field}_bjet_pt'], # rho is synonym for pt
                'phi': sample[f'nonRes_{field}_bjet_phi'],
                'eta': sample[f'nonRes_{field}_bjet_eta'],
                'tau': sample[f'nonRes_{field}_bjet_mass'], # tau is synonym for mass
            }, with_name='Momentum4D'
        )
    MbbReg_bjet_4moms = {}
    for field in ['lead', 'sublead']:
        MbbReg_bjet_4moms[f'{field}_bjet_4mom'] = ak.zip(
            {
                'rho': sample[f'nonResReg_{field}_bjet_pt'], # rho is synonym for pt
                'phi': sample[f'nonResReg_{field}_bjet_phi'],
                'eta': sample[f'nonResReg_{field}_bjet_eta'],
                'tau': sample[f'nonResReg_{field}_bjet_mass'], # tau is synonym for mass
            }, with_name='Momentum4D'
        )
    MbbRegDNNPair_bjet_4moms = {}
    for field in ['lead', 'sublead']:
        MbbRegDNNPair_bjet_4moms[f'{field}_bjet_4mom'] = ak.zip(
            {
                'rho': sample[f'nonResReg_DNNpair_{field}_bjet_pt'], # rho is synonym for pt
                'phi': sample[f'nonResReg_DNNpair_{field}_bjet_phi'],
                'eta': sample[f'nonResReg_DNNpair_{field}_bjet_eta'],
                'tau': sample[f'nonResReg_DNNpair_{field}_bjet_mass'], # tau is synonym for mass
            }, with_name='Momentum4D'
        )

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

    # Regressed dijet kinematics #
    dijet_4mom = bjet_4moms['lead_bjet_4mom'] + bjet_4moms['sublead_bjet_4mom']
    MbbReg_dijet_4mom = MbbReg_bjet_4moms['lead_bjet_4mom'] + MbbReg_bjet_4moms['sublead_bjet_4mom']
    MbbRegDNNPair_dijet_4mom = MbbRegDNNPair_bjet_4moms['lead_bjet_4mom'] + MbbRegDNNPair_bjet_4moms['sublead_bjet_4mom']

    # Nonres BDT variables #
    for field in ['lead', 'sublead']:
        # photon variables
        sample[f'{field}_sigmaE_over_E'] = sample[f'{field}_energyErr'] / (sample[f'{field}_pt'] * np.cosh(sample[f'{field}_eta']))
        
        # bjet variables
        sample[f'{field}_bjet_pt_over_Mjj'] = sample[f'nonRes_{field}_bjet_pt'] / sample['nonRes_dijet_mass']
        sample[f'MbbReg_{field}_bjet_pt_over_Mjj'] = sample[f'nonResReg_{field}_bjet_pt'] / sample['nonResReg_dijet_mass_DNNreg']
        sample[f'MbbRegDNNPair_{field}_bjet_pt_over_Mjj'] = sample[f'nonResReg_DNNpair_{field}_bjet_pt'] / sample['nonResReg_DNNpair_dijet_mass_DNNreg']
        
        sample[f'{field}_bjet_sigmapT_over_pT'] = sample[f'nonRes_{field}_bjet_PNetRegPtRawRes'] / sample[f'nonRes_{field}_bjet_pt']
        sample[f'MbbReg_{field}_bjet_sigmapT_over_pT'] = sample[f'nonResReg_{field}_bjet_PNetRegPtRawRes'] / sample[f'nonResReg_{field}_bjet_pt']
        sample[f'MbbRegDNNPair_{field}_bjet_sigmapT_over_pT'] = sample[f'nonResReg_DNNpair_{field}_bjet_PNetRegPtRawRes'] / sample[f'nonResReg_DNNpair_{field}_bjet_pt']


    # mHH variables #
    sample['pt_balance'] = sample['nonRes_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['nonRes_lead_bjet_pt'] + sample['nonRes_sublead_bjet_pt'])
    sample['MbbReg_pt_balance'] = sample['nonResReg_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['nonResReg_lead_bjet_pt'] + sample['nonResReg_sublead_bjet_pt'])
    sample['MbbRegDNNPair_pt_balance'] = sample['nonResReg_DNNpair_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['nonResReg_DNNpair_lead_bjet_pt'] + sample['nonResReg_DNNpair_sublead_bjet_pt'])


    # VH variables #
    sample['DeltaPhi_jj'] = deltaPhi(sample['nonRes_lead_bjet_phi'], sample['nonRes_sublead_bjet_phi'])
    sample['DeltaEta_jj'] = deltaEta(sample['nonRes_lead_bjet_eta'], sample['nonRes_sublead_bjet_eta'])
    sample['MbbReg_DeltaPhi_jj'] = deltaPhi(sample['nonResReg_lead_bjet_phi'], sample['nonResReg_sublead_bjet_phi'])
    sample['MbbReg_DeltaEta_jj'] = deltaEta(sample['nonResReg_lead_bjet_eta'], sample['nonResReg_sublead_bjet_eta'])
    sample['MbbRegDNNPair_DeltaPhi_jj'] = deltaPhi(sample['nonResReg_DNNpair_lead_bjet_phi'], sample['nonResReg_DNNpair_sublead_bjet_phi'])
    sample['MbbRegDNNPair_DeltaEta_jj'] = deltaEta(sample['nonResReg_DNNpair_lead_bjet_eta'], sample['nonResReg_DNNpair_sublead_bjet_eta'])

    # ISR-like jet variables
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample, dijet_4mom, jet_4moms)
    sample['isr_jet_pt'] = ak.where(isr_jet_bool, isr_jet_4mom.pt, FILL_VALUE)  # pt of isr jet
    sample['DeltaPhi_isr_jet_z'] = ak.where(  # phi angle between isr jet and z candidate
        isr_jet_bool,
        deltaPhi(isr_jet_4mom.phi, sample['nonRes_dijet_phi']), 
        FILL_VALUE
    )
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample, MbbReg_dijet_4mom, jet_4moms, prefactor='nonResReg')
    sample['MbbReg_isr_jet_pt'] = ak.where(isr_jet_bool, isr_jet_4mom.pt, FILL_VALUE)  # pt of isr jet
    sample['MbbReg_DeltaPhi_isr_jet_z'] = ak.where(  # phi angle between isr jet and z candidate
        isr_jet_bool,
        deltaPhi(isr_jet_4mom.phi, sample['nonResReg_dijet_phi']), 
        FILL_VALUE
    )
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample, MbbRegDNNPair_dijet_4mom, jet_4moms, prefactor='nonResReg_DNNpair')
    sample['MbbRegDNNPair_isr_jet_pt'] = ak.where(isr_jet_bool, isr_jet_4mom.pt, FILL_VALUE)  # pt of isr jet
    sample['MbbRegDNNPair_DeltaPhi_isr_jet_z'] = ak.where(  # phi angle between isr jet and z candidate
        isr_jet_bool,
        deltaPhi(isr_jet_4mom.phi, sample['nonResReg_DNNpair_dijet_phi']), 
        FILL_VALUE
    )

    # hash #
    sample['hash'] = np.arange(ak.num(sample['pt'], axis=0))  # Used to re-order the ttH killer output to match the input files

    # max non-bjet btag score -> sets lower limit for resampling #
    sample['max_nonbjet_btag'] = max_nonbjet_btag(sample)
    sample['MbbReg_max_nonbjet_btag'] = max_nonbjet_btag(sample, prefactor='nonResReg')
    sample['MbbRegDNNPair_max_nonbjet_btag'] = max_nonbjet_btag(sample, prefactor='nonResReg_DNNpair')
    
    sample['pass_mva-0.7'] = (
        (sample['lead_mvaID'] > -0.7)
        & (sample['sublead_mvaID'] > -0.7)
    )

def get_merged_filepath(unmerged_filepath, datasettype):
    merged_filepath = os.path.join(
        unmerged_filepath[:unmerged_filepath.rfind("Run3_202")+len("Run3_202x")] 
        + f"_mergedFull{datasettype}"
        + unmerged_filepath[unmerged_filepath.rfind("Run3_202")+len("Run3_202x"):],
        ""
    )
    if (
        re.search("nominal", merged_filepath) is not None
        and re.search("data", merged_filepath) is not None
    ):
        merged_filepath = merged_filepath[:merged_filepath.find('nominal')]
        
    return merged_filepath


def slim_parquets(sample, datasettype):
    if datasettype == 'Resolved':
        return slim_parquets_resolved(sample)
    elif datasettype == 'Boosted':
        return slim_parquets_boosted(sample)
    else:
        raise NotImplementedError(f"Datasettype you requested ({datasettype}) is not implemented. We only have implemented: {DATASETTYPE.keys()}")

def slim_parquets_resolved(sample):
    sample_fields = [field for field in sample.fields]
    for field in sample.fields:
        if (
            re.match('Res', field) is not None  # applies event cut on dijet mass
            or re.search('VBF', field) is not None  # ignoring VBF category for now...
            or re.match('fatjet', field) is not None  # we don't care about ak8 jets
        ):
            sample_fields.remove(field)
    
    return ak.zip({
        field: sample[field] for field in sample_fields
    }, depth_limit=1)

def slim_parquets_boosted(sample):
    sample_fields = [field for field in sample.fields]
    for field in sample.fields:
        if (
            re.match('nonRes', field) is not None  # applies event cut on dijet mass, but we want fatjet
            or re.search('VBF', field) is not None  # ignoring VBF category for now...
            or re.match('jet', field) is not None  # we don't care about ak4 jets
            or re.search('dijet', field) is not None  # we don't want dijet, want fatjet as H->bb object
            or re.search('DeltaR_j', field) is not None  # we don't care about ak4 jets
            or re.search('DeltaR_b', field) is not None  # we don't care about ak4 jets
            # or re.search('lepton', field) is not None  # we don't want leptons
        ):
            sample_fields.remove(field)
    return ak.zip({
        field: sample[field] for field in sample_fields
    }, depth_limit=1)

# function taken from HiggsDNA btagSF functions, implemented by Manos I believe...
def correct_weights(sample, sample_filepath_list, computebtag=True):
    sumGenWeightsBeforePresel = 0.  # sum of gen weights before pre-selection (for proper scaling of genweights)
    sumWeightCentral, sumWeightCentralNoBtagSF = 0., 0.  # sum of central weights before and after bTag SF (for renormalizing btag SFs)
    sumWeightBtagSys = {
        # correlated across years
        'lfUp': 0, 'lfDown': 0, 'hfUp': 0, 'hfDown': 0,
        # uncorrelated across years
        'lfstats1Up': 0, 'lfstats1Down': 0, 'lfstats2Up': 0, 'lfstats2Down': 0, 
        'hfstats1Up': 0, 'hfstats1Down': 0, 'hfstats2Up': 0, 'hfstats2Down': 0, 
        # not currently necessary for btag_sys but computed anyways just in case
        'jesUp': 0, 'jesDown': 0, 'cferr1Up': 0, 'cferr1Down': 0, 'cferr2Up': 0,'cferr2Down': 0
    }

    for sample_filepath in sample_filepath_list:
        # presel #
        sumGenWeightsBeforePresel += float(pq.read_table(sample_filepath).schema.metadata[b'sum_genw_presel'])
        # btag SF #
        sumWeightCentral += float(pq.read_table(sample_filepath).schema.metadata[b'sum_weight_central'])
        sumWeightCentralNoBtagSF += float(pq.read_table(sample_filepath).schema.metadata[b'sum_weight_central_wo_bTagSF'])
        if computebtag:
            for btagsys in sumWeightBtagSys.keys():
                sumWeightBtagSys[btagsys] += float(
                    pq.read_table(sample_filepath).schema.metadata[bytes(f"sum_weight_bTagSF_sys_{btagsys}", encoding='utf8')]
                )

    # Rescale weights by sum of genweights
    sample['weight_nominal'] = sample['weight']
    syst_weight_fields = [field for field in sample.fields if (("weight_" in field) and ("Up" in field or "Down" in field))]
    for weight_field in ["weight"] + syst_weight_fields:
        sample[weight_field] = sample[weight_field] / sumGenWeightsBeforePresel

    # Rescale bTag SF weights by preBTag SF sum to get correct normalization
    sample['weight'] = sample['weight'] * (sumWeightCentralNoBtagSF / sumWeightCentral)
    print(f"central weight correction ratio (pre-bTag SF integral / post-bTag SF integral) = {sumWeightCentralNoBtagSF / sumWeightCentral}")
    for btagsys in sumWeightBtagSys.keys():
        if sumWeightBtagSys[btagsys] != 0:
            print(f"{btagsys} weight correction ratio (pre-{btagsys} integral / post-{btagsys} integral) = {sumWeightCentralNoBtagSF / sumWeightBtagSys[btagsys]}")
            sample[f"weight_bTagSF_sys_{btagsys}"] = sample[f"weight_bTagSF_sys_{btagsys}"] * (
                sumWeightCentralNoBtagSF / sumWeightBtagSys[btagsys]
            )
    

def main():
    sim_dir_lists = {
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "preEE", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "postEE", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "preBPix", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "postBPix", ""): None,

        # os.path.join(lpc_fileprefix, "Run3_2022_SMEFTSingleH", "2022postEE", ""): None,
        # os.path.join(lpc_fileprefix, "Run3_2022_SMEFTSignal", "2022postEE", ""): None,
        # os.path.join(lpc_fileprefix, "Run3_2022_ggHH_smeft", "2022postEE", ""): None,
        # os.path.join(lpc_fileprefix, "Run3_2022_ggH_smeft", "2022postEE", ""): None,
        
    }
    data_dir_lists = {
        os.path.join(lpc_fileprefix, "Run3_2022", "data", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "data", ""): None,
        # os.path.join(lpc_fileprefix, "Run3_2024", "data", ""): None,
        # os.path.join(lpc_fileprefix, "Run3_2024_temp_v14", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2024_v14_JECs_vetomaps", ""): None,
    }
    
    # MC Era: total era luminosity [fb^-1] #
    luminosities = {
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "preEE", ""): 7.9804,
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "postEE", ""): 26.6717,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "preBPix", ""): 17.794,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "postBPix", ""): 9.451,
        os.path.join(lpc_fileprefix, "Run3_2024", "sim", "2024", ""): 109.08,

        # os.path.join(lpc_fileprefix, "Run3_2022_SMEFTSingleH", "2022postEE", ""): 26.6717,
        # os.path.join(lpc_fileprefix, "Run3_2022_SMEFTSignal", "2022postEE", ""): 26.6717,
        # os.path.join(lpc_fileprefix, "Run3_2022_ggHH_smeft", "2022postEE", ""): 26.6717,
        # os.path.join(lpc_fileprefix, "Run3_2022_ggH_smeft", "2022postEE", ""): 26.6717,
    }
    
    
    # Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
    #   -> Do we not need to care about other HH processes? https://arxiv.org/pdf/1910.00012.pdf
    cross_sections = {
        # signal #
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_HH_c
        'GluGluToHH': 34.43*0.0026,
        # 'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': 34.43*0.0026,
        # 'GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00': 34.43*0.0026,
        'GluGlutoHH_kl-1p00_kt-1p00_c2-0p00': 34.43*0.0026,

        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_di_H
        # 'VBFHHto2B2G_CV_1_C2V_1_C3_1': 1.870*0.0026,

        # non-resonant background #
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGG-Box-3Jets_MGG-80_13p6TeV_sherpa
        'GGJets': 88750., 
        'GGJets_MGG-40to80': 318100.,
        'GGJets_MGG-80': 88750.,
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-20to40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt20To40': 242500., 
        'GJet_PT-20to40_DoubleEMEnriched_MGG-80': 242500., 
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt40': 919100., 
        'GJet_PT-40_DoubleEMEnriched_MGG-80': 919100., 

        # resonant background #
        #    -> xs taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap#ggF_N3LO_QCD_NLO_EW
        # ggF H
        'GluGluHToGG': 52170*0.00228,
        'GluGluHToGG_M_125': 52170*0.00228,
        'GluGluHtoGG': 52170*0.00228,
        # ttH
        'ttHToGG': 568.8*0.00228,
        'ttHtoGG_M_125': 568.8*0.00228,
        'ttHtoGG': 568.8*0.00228,
        # VBF H
        'VBFHToGG': 4075*0.00228,
        'VBFHToGG_M_125': 4075*0.00228,
        'VBFHtoGG': 4075*0.00228,
        # VH
        'VHToGG': (1453 + 942.2)*0.00228,
        'VHtoGG_M_125': (1453 + 942.2)*0.00228,
        'VHtoGG': (1453 + 942.2)*0.00228,
        'VHtoGG_M-125': (1453 + 942.2)*0.00228,
        # bbH
        'BBHto2G_M_125': 525.1*0.00228,
        'bbHtoGG': 525.1*0.00228,

        # kappa lambda scane signal samples #
        'GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl-0p00_kt-1p00_c2-0p00': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl-2p45_kt-1p00_c2-0p00': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl-5p00_kt-1p00_c2-0p00': 34.43*0.0026,

        # extra VH samples (produced by Irene) #
        'ZH_sample': 942.2*0.00228*0.69911,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppZH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2018/listings/rpp2018-list-z-boson.pdf
        'ZH_Hto2G_Zto2Q_M-125': 942.2*0.00228*0.69911,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppWH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2022/listings/rpp2022-list-w-boson.pdf
        'WminusH_Hto2G_Wto2Q_M-125': 1453*0.00228*0.6741,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppWH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2022/listings/rpp2022-list-w-boson.pdf
        'WplusH_Hto2G_Wto2Q_M-125': 1453*0.00228*0.6741,
        
        # Other potential background samples
        'DDQCDGJets': 1,
        'TTGG': 16.96,
        # 'TTGJetPt10To100': 4216.,
        # 'TTG_1Jets_PTG_10to100': 4216.,
        # 'TTGJetPt100To200': 411.4,
        # 'TTG_1Jets_PTG_100to200': 411.4,
        'TTG-1Jets_PTG-100to200': 411.4,
        # 'TTGJetPt200': 128.4,
        # 'TTG_1Jets_PTG_200': 128.4,
        'TTG-1Jets_PTG-200': 128.4,

        # # SMEFT samples #
        # # singleH(?) background
        # 'GluGluToHH': 1, 'STXS': 1, 'bbh': 1, 'tth': 1, 'vbf': 1, 'vh': 1,
        # # signal
        # 'GluGluToHH': 1, 'ggHH-CH-20-CHD10-t1': 1, 'ggHH-CH-20-CHG0.1-t1': 1, 'ggHH-CH-20-CHbox20-t1': 1,
        # 'ggHH-CH-20-CuH40-t1': 1, 'ggHH-CH-20-t1': 1, 'ggHH-CH-6-t1': 1, 'ggHH-CH10-t1': 1, 'ggHH-CHD-5-t1': 1, 
        # 'ggHH-CHD10-CHG0.1-t1': 1, 'ggHH-CHD10-CuH40-t1': 1, 'ggHH-CHD10-t1': 1, 'ggHH-CHG-0.05-t1': 1,
        # 'ggHH-CHG0.1-t1': 1, 'ggHH-CHbox-10-t1': 1, 'ggHH-CHbox20-CHD10-t1': 1, 'ggHH-CHbox20-CHG0.1-t1': 1,
        # 'ggHH-CHbox20-CuH40-t1': 1, 'ggHH-CHbox20-t1': 1, 'ggHH-CuH-20-t1': 1, 'ggHH-CuH40-CHG0.1-t1': 1,
        # 'ggHH-CuH40-t1': 1, 'ggHH_BM1': 1, 'ggHH_BM3': 1, 'ggHH_kl_0p00': 1, 'ggHH_kl_2p45': 1, 'ggHH_kl_5p00': 1,
    }
    sample_name_map = {
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': 'GluGluToHH', 
        'GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00': 'GluGluToHH', 
        'GluGlutoHH_kl-1p00_kt-1p00_c2-0p00': 'GluGluToHH',

        'GluGluHToGG_M_125': 'GluGluHToGG',
        'GluGluHtoGG': 'GluGluHToGG',

        'ttHtoGG_M_125': 'ttHToGG', 
        'ttHtoGG': 'ttHToGG',

        'VBFHToGG_M_125': 'VBFHToGG', 
        'VBFHtoGG': 'VBFHToGG', 
        
        'VHtoGG_M_125': 'VHToGG',
        'VHtoGG': 'VHToGG',
        'VHtoGG_M-125': 'VHToGG',

        'BBHto2G_M_125': 'bbHToGG', 
        'bbHtoGG': 'bbHToGG',

        'ZH_Hto2G_Zto2Q_M-125': 'ZHToQQGG', 
        'WminusH_Hto2G_Wto2Q_M-125': 'W-HToQQGG', 
        'WplusH_Hto2G_Wto2Q_M-125': 'W+HToQQGG',

        'VBFHHto2B2G_CV_1_C2V_1_C3_1': 'VBFToHH',

        'TTG_1Jets_PTG_10to100': 'TTGJetPt10To100',
        'TTG-1Jets_PTG-100to200': 'TTGJetPt100To200',
        'TTG-1Jets_PTG-200': 'TTGJetPt200',

        'GGJets_MGG-40to80': 'GGJets',
        'GGJets_MGG-80': 'GGJets',
        'GJet_PT-20to40_DoubleEMEnriched_MGG-80': 'GJetPt20To40',
        'GJet_PT-40_DoubleEMEnriched_MGG-80': 'GJetPt40'
    }
    
    # Pull MC sample dir_list
    for sim_era in sim_dir_lists.keys():
        all_sim_dirs_set = set(os.listdir(sim_era))

        cross_sections_set = set([key for key in cross_sections.keys()])

        if not FORCE_RERUN:
            try:
                already_run_dirs_set = set()
                for datasettype in DATASETTYPE:
                    if len(already_run_dirs_set) == 0:
                        already_run_dirs_set = set(os.listdir(get_merged_filepath(sim_era, datasettype)))
                        continue
                    already_run_dirs_set = already_run_dirs_set & set(os.listdir(get_merged_filepath(sim_era, datasettype)))
            except:
                FileNotFoundError
                already_run_dirs_set = set()
        else:
            already_run_dirs_set = set()

        sim_dir_lists[sim_era] = list(
            (all_sim_dirs_set & cross_sections_set) - already_run_dirs_set
        )
        sim_dir_lists[sim_era].sort()

    # Pull Data sample dir_list
    for data_era in data_dir_lists.keys():
        all_data_dirs_set = set(os.listdir(data_era))

        bad_dirs_set = set()
        for data_dir in all_data_dirs_set:
            if data_dir[0] == '.':
                bad_dirs_set.add(data_dir)

        if not FORCE_RERUN:
            try:
                already_run_dirs_set = set()
                for datasettype in DATASETTYPE:
                    if len(already_run_dirs_set) == 0:
                        already_run_dirs_set = set(os.listdir(get_merged_filepath(sim_era, datasettype)))
                        continue
                    already_run_dirs_set = already_run_dirs_set & set(os.listdir(get_merged_filepath(sim_era, datasettype)))
            except:
                FileNotFoundError
                already_run_dirs_set = set()
        else:
            already_run_dirs_set = set()

        data_dir_lists[data_era] = list(
            (all_data_dirs_set - bad_dirs_set) - already_run_dirs_set
        )
        data_dir_lists[data_era].sort()
        

    # Perform the variable calculation and merging
    for sim_era, dir_list in sim_dir_lists.items():

        for dir_name in dir_list:
            sample_dirpath = os.path.join(sim_era, dir_name, "")

            for sample_type in os.listdir(sample_dirpath):

                # if sample_type != 'nominal': continue
                # if re.search('HH', dir_name) is None or re.search('preBPix', sim_era) is not None: continue
                if sample_type != 'nominal' and re.search('H', dir_name) is None: continue
                if re.search('VHtoGG', dir_name) is None or re.search('preBPix', sim_era) is None or sample_type != 'nominal': continue

                print(sim_era[sim_era[:-1].rfind('/')+1:-1]+f': {dir_name} - {sample_type}')
                if re.search('.parquet', sample_type) is not None:
                    sample_type_dirpath = os.path.join(sample_dirpath, "")
                else:
                    sample_type_dirpath = os.path.join(sample_dirpath, sample_type, "")

                # Load all the parquets of a single sample into an ak array
                sample_filepath_list = glob.glob(os.path.join(sample_type_dirpath, '*.parquet'))
                sample_list = [ak.from_parquet(file) for file in sample_filepath_list]
                if len(sample_list) < 1:
                    print("No files to concatenate. Skipping sample.")
                    continue
                sample = ak.concatenate(sample_list)

                if 'weight_nominal' not in sample.fields and dir_name != 'DDQCDGJets':
                    correct_weights(
                        sample, sample_filepath_list, 
                        computebtag=(sample_type=='nominal')
                    )

                for datasettype in DATASETTYPE:
                    # Slim parquets by removing Res fields (for now)
                    slim_sample = slim_parquets(sample, datasettype)

                    # Add useful parquet meta-info
                    slim_sample['sample_name'] = dir_name if dir_name not in sample_name_map else sample_name_map[dir_name]
                    slim_sample['sample_era'] = sim_era[sim_era[:-1].rfind('/')+1:-1]
                    slim_sample['eventWeight'] = slim_sample['weight'] * luminosities[sim_era] * cross_sections[dir_name]

                    # Add necessary extra variables
                    add_vars(slim_sample, datasettype)
            
                    # Save out merged parquet
                    destdir = get_merged_filepath(sample_type_dirpath, datasettype=datasettype)
                    if not os.path.exists(destdir):
                        os.makedirs(destdir)
                    filepath = os.path.join(destdir, dir_name+'_merged.parquet')
                    merged_parquet = ak.to_parquet(slim_sample, filepath)
                    del slim_sample
                    print('======================== \n', destdir)
                
                # Delete sample for memory reasons
                del sample

    # for data_era, dir_list in data_dir_lists.items():

    #     for dir_name in dir_list:

    #         if re.search('.parquet', dir_name) is not None:
    #             sample_dirpath = os.path.join(data_era, "")
    #         else:
    #             sample_dirpath = os.path.join(data_era, dir_name, "")

    #         # Load all the parquets of a single sample into an ak array
    #         print(dir_name)
    #         sample_list = []
    #         for file in glob.glob(os.path.join(sample_dirpath, '*.parquet')):
    #             try:
    #                 sample_list.append(ak.from_parquet(file))
    #             except:
    #                 print('append failed, skipping file')
    #                 continue
    #         if len(sample_list) < 1:
    #             print("No files to concatenate. Skipping sample.")
    #             continue
    #         sample = ak.concatenate(sample_list)
    #         print(ak.size(sample, axis=0))

    #         for datasettype in DATASETTYPE:
    #             # Slim parquets by removing Res fields (for now)
    #             slim_sample = slim_parquets(sample, datasettype)

    #             wanted_fields = {
    #                 'dZ', 
    #                 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90',  # old triggers
    #                 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95',
    #                 'DiphotonMVA14p25_Mass90',  # new triggers
    #                 'DiphotonMVA14p25_Tight_Mass90'
    #             }
    #             for field in wanted_fields:
    #                 if field in slim_sample.fields: continue
    #                 elif field not in sample.fields: continue
    #                 slim_sample[field] = sample[field]

    #             # Add useful parquet meta-info
    #             slim_sample['sample_name'] = dir_name
    #             slim_sample['sample_era'] = data_era[data_era[:-1].rfind('/')+1:-1]

    #             # Add necessary extra variables
    #             add_vars(slim_sample, datasettype)
        
    #             # Save out merged parquet
    #             destdir = get_merged_filepath(sample_dirpath, datasettype=datasettype)
    #             if not os.path.exists(destdir):
    #                 os.makedirs(destdir)
    #             if re.search('.parquet', dir_name) is None:
    #                 filepath = os.path.join(destdir, dir_name+'_merged.parquet')
    #             else:
    #                 filepath = os.path.join(destdir, dir_name[:dir_name.find('.parquet')]+'_merged'+dir_name[dir_name.find('.parquet'):])
    #             merged_parquet = ak.to_parquet(slim_sample, filepath)
    #             del slim_sample
    #             print('======================== \n', destdir)
            
    #         # Delete sample for memory reasons
    #         del sample


if __name__ == '__main__':
    main()
