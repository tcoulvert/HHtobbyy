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

# need to delete the extra files under /store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/Run3_2023/ preBPix and postBPix

# lpc_redirector = "root://cmseos.fnal.gov/"
# lxplus_redirector = "root://eosuser.cern.ch/"
# lxplus_fileprefix = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v2"
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/"
FILL_VALUE = -999
NUM_JETS = 10
FORCE_RERUN = True


def add_vars(sample, data=False):
    
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
    
    def jet_mask(sample, i):
        return (
            ak.where(
                sample['nonRes_lead_bjet_jet_idx'] != i, True, False
            ) & ak.where(
                sample['nonRes_sublead_bjet_jet_idx'] != i, True, False
            ) & ak.where(sample[f'jet{i}_mass'] != FILL_VALUE, True, False)
        )

    def zh_isr_jet(sample, dijet_4mom, jet_4moms):
        min_total_pt = ak.Array([FILL_VALUE for _ in range(ak.num(sample['event'], axis=0))])
        isr_jet_4mom = copy.deepcopy(jet_4moms['jet1_4mom'])

        for i in range(1, NUM_JETS+1):
            jet_i_mask = jet_mask(sample, i)

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
    
    def max_nonbjet_btag(sample):
        max_btag_score = ak.Array([0. for _ in range(ak.num(sample['event'], axis=0))])

        for i in range(1, NUM_JETS+1):
            jet_i_mask = jet_mask(sample, i)

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
                'rho': sample[f'nonRes_{field}_bjet_pt'] * sample[f'nonRes_{field}_bjet_PNetRegPtRawCorr'] * sample[f'nonRes_{field}_bjet_PNetRegPtRawCorrNeutrino'], # rho is synonym for pt
                'phi': sample[f'nonRes_{field}_bjet_phi'],
                'eta': sample[f'nonRes_{field}_bjet_eta'],
                'tau': sample[f'nonRes_{field}_bjet_mass'], # tau is synonym for mass
            }, with_name='Momentum4D'
        )
    # Improved bjet bTag score and regressed mass #
    for field in ['lead', 'sublead']:
        sample[f'{field}_bjet_btagRobustParTAK4B'] = robustParT(sample, bjet_type=field)
        sample[f'{field}_bjet_PNetRegPt'] = bjet_4moms[f'{field}_bjet_4mom'].pt
        sample[f'{field}_bjet_sigmapT_over_pT'] = sample[f'nonRes_{field}_bjet_PNetRegPtRawRes'] / sample[f'nonRes_{field}_bjet_pt']
        sample[f'{field}_bjet_sigmapT_over_RegPt'] = sample[f'nonRes_{field}_bjet_PNetRegPtRawRes'] / sample[f'{field}_bjet_PNetRegPt']

    # Regressed jet kinematics #
    jet_4moms = {}
    for i in range(1, NUM_JETS+1):
        jet_4moms[f'jet{i}_4mom'] = ak.zip(
            {
                'rho': sample[f'jet{i}_pt'] * sample[f'jet{i}_PNetRegPtRawCorr'] * sample[f'jet{i}_PNetRegPtRawCorrNeutrino'],
                'phi': sample[f'jet{i}_phi'],
                'eta': sample[f'jet{i}_eta'],
                'tau': sample[f'jet{i}_mass'],
            }, with_name='Momentum4D'
        )

    # Regressed dijet kinematics #
    dijet_4mom = bjet_4moms['lead_bjet_4mom'] + bjet_4moms['sublead_bjet_4mom']
    sample['dijet_PNetRegPt'] = dijet_4mom.pt
    sample['dijet_PNetRegEta'] = dijet_4mom.eta
    sample['dijet_PNetRegPhi'] = dijet_4mom.phi
    sample['dijet_PNetRegMass'] = dijet_4mom.mass

    # Regressed HH kinematics
    diphoton_4mom = ak.zip(
        {
            'rho': sample['pt'],
            'phi': sample['phi'],
            'eta': sample['eta'],
            'tau': sample['mass'],
        }, with_name='Momentum4D'
    )
    HH_4mom = diphoton_4mom + dijet_4mom
    sample['HH_PNetRegPt'] = HH_4mom.pt
    sample['HH_PNetRegEta'] = HH_4mom.eta
    sample['HH_PNetRegPhi'] = HH_4mom.phi
    sample['HH_PNetRegMass'] = HH_4mom.mass

    # Nonres BDT variables #
    for field in ['lead', 'sublead']:
        # photon variables
        sample[f'{field}_sigmaE_over_E'] = sample[f'{field}_energyErr'] / (sample[f'{field}_pt'] * np.cosh(sample[f'{field}_eta']))
        # bjet variables
        sample[f'{field}_bjet_pt_over_Mjj'] = sample[f'nonRes_{field}_bjet_pt'] / sample['nonRes_dijet_mass']
        sample[f'{field}_bjet_RegPt_over_Mjj'] = sample[f'{field}_bjet_PNetRegPt'] / sample['dijet_PNetRegMass']

    # mHH variables #
    sample['RegPt_balance'] = sample['HH_PNetRegPt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['lead_bjet_PNetRegPt'] + sample['sublead_bjet_PNetRegPt'])
    sample['pt_balance'] = sample['nonRes_HHbbggCandidate_pt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['nonRes_lead_bjet_pt'] + sample['nonRes_sublead_bjet_pt'])


    # VH variables #
    sample['DeltaPhi_jj'] = deltaPhi(sample['nonRes_lead_bjet_phi'], sample['nonRes_sublead_bjet_phi'])
    sample['DeltaEta_jj'] = deltaEta(sample['nonRes_lead_bjet_eta'], sample['nonRes_sublead_bjet_eta'])
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample, dijet_4mom, jet_4moms)
    sample['isr_jet_RegPt'] = ak.where(isr_jet_bool, isr_jet_4mom.pt, FILL_VALUE)  # pt of isr jet
    sample['DeltaPhi_isr_jet_z'] = ak.where(  # phi angle between isr jet and z candidate
        isr_jet_bool,
        deltaPhi(isr_jet_4mom.phi, sample['nonRes_dijet_phi']), 
        FILL_VALUE
    )

    # hash #
    hash_arr = np.zeros_like(ak.to_numpy(sample['pt']))
    for event_idx in range(len(sample['pt'])):
        hash_arr[event_idx] = hash(str(sample['event'])+str(sample['lumi'])+str(sample['run']))
    sample['hash'] = hash_arr  # Used to re-order the ttH killer output to match the input files

    # max non-bjet btag score -> sets lower limit for resampling #
    sample['max_nonbjet_btag'] = max_nonbjet_btag(sample)


def get_merged_filepath(unmerged_filepath):
    return os.path.join(
        unmerged_filepath[:unmerged_filepath.rfind("Run3_202")+len("Run3_202x")] 
        + "_merged"
        + unmerged_filepath[unmerged_filepath.rfind("Run3_202")+len("Run3_202x"):],
        ""
    )

def slim_parquets(sample):
    sample_fields = [field for field in sample.fields]
    for field in sample.fields:
        if re.match('Res', field) is not None or re.search('4mom', field) is not None:
            sample_fields.remove(field)
    sample = ak.zip({
        field: sample[field] for field in sample_fields
    })

def main():
    sim_dir_lists = {
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "preEE", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "postEE", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "preBPix", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "postBPix", ""): None,
    }
    data_dir_lists = {
        os.path.join(lpc_fileprefix, "Run3_2022", "data", ""): None,
        os.path.join(lpc_fileprefix, "Run3_2023", "data", ""): None,
    }
    
    # MC Era: total era luminosity [fb^-1] #
    luminosities = {
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "preEE", ""): 7.9804,
        os.path.join(lpc_fileprefix, "Run3_2022", "sim", "postEE", ""): 26.6717,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "preBPix", ""): 17.794,
        os.path.join(lpc_fileprefix, "Run3_2023", "sim", "postBPix", ""): 9.451,
        os.path.join(lpc_fileprefix, "Run3_2024", "sim", "2024", ""): 109.08,
    }

    
    
    # Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
    #   -> Do we not need to care about other HH processes? https://arxiv.org/pdf/1910.00012.pdf
    cross_sections = {
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_HH_c
        'GluGluToHH': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': 34.43*0.0026,
        'GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00': 34.43*0.0026,
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGG-Box-3Jets_MGG-80_13p6TeV_sherpa
        'GGJets': 88750, 
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-20to40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt20To40': 242500, 
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt40': 919100, 
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#gluon_gluon_Fusion_Process
        'GluGluHToGG': 48520*0.00228,
        'GluGluHToGG_M_125': 48520*0.00228,
        'GluGluHtoGG': 48520*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ttH_Process
        'ttHToGG': 506.5*0.00228,
        'ttHtoGG_M_125': 506.5*0.00228,
        'ttHtoGG': 506.5*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#VBF_Process
        'VBFHToGG': 3779*0.00228,
        'VBFHToGG_M_125': 3779*0.00228,
        'VBFHtoGG': 3779*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#WH_Process + https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
        'VHToGG': (1369 + 882.4)*0.00228,
        'VHtoGG_M_125': (1369 + 882.4)*0.00228,
        'VHtoGG': (1369 + 882.4)*0.00228,
        'VHtoGG_M-125': (1369 + 882.4)*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#bbH_Process
        'BBHto2G_M_125': 526.5*0.00228,
        'bbHtoGG': 526.5*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppZH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2018/listings/rpp2018-list-z-boson.pdf
        'ZH_Hto2G_Zto2Q_M-125': 882.4*0.00228*0.69911,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppWH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2022/listings/rpp2022-list-w-boson.pdf
        'WminusH_Hto2G_Wto2Q_M-125': 1369*0.00228*0.6741,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ppWH_Total_Cross_Section_with_ap +  https://pdg.lbl.gov/2022/listings/rpp2022-list-w-boson.pdf
        'WplusH_Hto2G_Wto2Q_M-125': 1369*0.00228*0.6741,
        # Other potential samples
        'DDQCDGJets': 1,
        'TTGG': 1
    }
    sample_name_map = {
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': 'GluGluToHH', 
        'GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00': 'GluGluToHH', 
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
        'WplusH_Hto2G_Wto2Q_M-125': 'W+HToQQGG'
    }
    
    # Pull MC sample dir_list
    for sim_era in sim_dir_lists.keys():
        all_sim_dirs_set = set(os.listdir(sim_era))

        cross_sections_set = set([key for key in cross_sections.keys()])

        if not FORCE_RERUN:
            try:
                already_run_dirs_set = set(
                    os.listdir(get_merged_filepath(sim_era))
                )
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
                already_run_dirs_set = set(
                    os.listdir(get_merged_filepath(data_era))
                )
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

                sample_type_dirpath = os.path.join(sample_dirpath, sample_type, "")

                # Load all the parquets of a single sample into an ak array
                print(sim_era[sim_era[:-1].rfind('/')+1:-1]+': '+dir_name)
                sample_list = [ak.from_parquet(file) for file in glob.glob(os.path.join(sample_type_dirpath, '*.parquet'))]
                if len(sample_list) < 1:
                    continue
                sample = ak.concatenate(sample_list)

                if 'weight_nominal' not in sample.fields and dir_name != 'DDQCDGJets':
                    # Compute sum of gen weights
                    sample['sumGenWeights'] = sum(
                        float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in glob.glob(
                            os.path.join(sample_type_dirpath, '*.parquet')
                        )
                    )
                    # Rescale weights by sum of genweights
                    sample['weight_nominal'] = sample['weight']
                    syst_weight_fields = [field for field in sample.fields if (("weight_" in field) and ("Up" in field or "Down" in field))]
                    for weight_field in ["weight"] + syst_weight_fields:
                        sample[weight_field] = sample[weight_field] / sample['sumGenWeights']

                # Slim parquets by removing Res fields (for now)
                slim_parquets(sample)

                # Add useful parquet meta-info
                sample['sample_name'] = dir_name if dir_name not in sample_name_map else sample_name_map[dir_name]
                sample['sample_era'] = sim_era[sim_era[:-1].rfind('/')+1:-1]
                sample['eventWeight'] = sample['weight'] * luminosities[sim_era] * cross_sections[dir_name]

                # Add necessary extra variables
                add_vars(sample)
        
                # Save out merged parquet
                destdir = get_merged_filepath(sample_type_dirpath)
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                filepath = os.path.join(destdir, dir_name+'_merged.parquet')
                merged_parquet = ak.to_parquet(sample, filepath)
                
                # Delete sample for memory reasons
                del sample
                print('======================== \n', destdir)

    # for data_era, dir_list in data_dir_lists.items():

    #     for dir_name in dir_list:

    #         sample_dirpath = os.path.join(data_era, dir_name, "")

    #         # Load all the parquets of a single sample into an ak array
    #         print(dir_name)
    #         sample_list = [ak.from_parquet(file) for file in glob.glob(os.path.join(sample_dirpath, '*.parquet'))]
    #         if len(sample_list) < 1:
    #             continue
    #         sample = ak.concatenate(sample_list)

    #         # Slim parquets by removing Res fields (for now)
    #         slim_parquets(sample)
            
    #         # Add useful parquet meta-info
    #         sample['sample_name'] = dir_name
    #         sample['sample_era'] = data_era[data_era[:-1].rfind('/')+1:-1]

    #         # Add necessary extra variables
    #         add_vars(sample, data=True)
    
    #         # Save out merged parquet
    #         destdir = get_merged_filepath(sample_dirpath)
    #         if not os.path.exists(destdir):
    #             os.makedirs(destdir)
    #         filepath = os.path.join(destdir, dir_name+'_merged.parquet')
    #         merged_parquet = ak.to_parquet(sample, filepath)
            
    #         # Delete sample for memory reasons
    #         del sample
    #         print('======================== \n', destdir)


if __name__ == '__main__':
    main()
