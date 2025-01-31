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
LPC_FILEPREFIX1 = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/Run3_2022_merged_v1"
LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/Run3_2022"
FILL_VALUE = -999
NUM_JETS = 10


def add_vars(sample):
    
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
            ) & ak.where(sample[f'jet{i}_pt'] != FILL_VALUE, True, False)
        )

    def zh_isr_jet(sample):
        min_total_pt = ak.Array([FILL_VALUE for _ in range(ak.num(sample['event'], axis=0))])
        isr_jet_4mom = copy.deepcopy(sample['jet1_4mom'])

        for i in range(1, NUM_JETS+1):
            jet_i_mask = jet_mask(sample, i)

            z_jet_4mom = sample['dijet_4mom'] + sample[f'jet{i}_4mom']

            better_isr_bool = (
                ak.where(z_jet_4mom.pt < min_total_pt, True, False)
                | ak.where(min_total_pt == FILL_VALUE, True, False)
            ) & jet_i_mask
            min_total_pt = ak.where(
                better_isr_bool, z_jet_4mom.pt, min_total_pt
            )
            isr_jet_4mom = ak.where(
                better_isr_bool, sample[f'jet{i}_4mom'], isr_jet_4mom
            )
        return isr_jet_4mom, ak.where(min_total_pt != FILL_VALUE, True, False)
    
    def robustParT(sample, bjet_type='lead'):
        robust_parT = ak.Array([FILL_VALUE for _ in range(ak.num(sample['event'], axis=0))])

        for i in range(1, NUM_JETS+1):
            robust_parT = ak.where(sample[f"nonRes_{bjet_type}_bjet_jet_idx"] == i, sample[f"jet{i}_btagRobustParTAK4B"], robust_parT)

        return robust_parT
    
    # Regressed bjet kinematics #
    for field in ['lead', 'sublead']:
        sample[f'{field}_bjet_4mom'] = ak.zip(
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
        sample[f'{field}_bjet_PNetRegPt'] = sample[f'{field}_bjet_4mom'].pt

    # Regressed jet kinematics #
    for i in range(1, NUM_JETS+1):
        sample[f'jet{i}_4mom'] = ak.zip(
            {
                'rho': sample[f'jet{i}_pt'] * sample[f'jet{i}_PNetRegPtRawCorr'] * sample[f'jet{i}_PNetRegPtRawCorrNeutrino'],
                'phi': sample[f'jet{i}_phi'],
                'eta': sample[f'jet{i}_eta'],
                'tau': sample[f'jet{i}_mass'],
            }, with_name='Momentum4D'
        )

    # Regressed dijet kinematics #
    sample['dijet_4mom'] = sample['lead_bjet_4mom'] + sample['sublead_bjet_4mom']
    sample['dijet_PNetRegPt'] = sample['dijet_4mom'].pt
    sample['dijet_PNetRegEta'] = sample['dijet_4mom'].eta
    sample['dijet_PNetRegPhi'] = sample['dijet_4mom'].phi
    sample['dijet_PNetRegMass'] = sample['dijet_4mom'].mass

    # Regressed HH kinematics
    sample['diphoton_4mom'] = ak.zip(
        {
            'rho': sample['pt'],
            'phi': sample['phi'],
            'eta': sample['eta'],
            'tau': sample['mass'],
        }, with_name='Momentum4D'
    )
    sample['HH_4mom'] = sample['diphoton_4mom'] + sample['dijet_4mom']
    sample['HH_PNetRegPt'] = sample['HH_4mom'].pt
    sample['HH_PNetRegEta'] = sample['HH_4mom'].eta
    sample['HH_PNetRegPhi'] = sample['HH_4mom'].phi
    sample['HH_PNetRegMass'] = sample['HH_4mom'].mass

    # Nonres BDT variables #
    for field in ['lead', 'sublead']:
        # photon variables
        sample[f'{field}_sigmaE_over_E'] = sample[f'{field}_energyErr'] / (sample[f'{field}_pt'] * np.cosh(sample[f'{field}_eta']))
        # bjet variables
        sample[f'{field}_bjet_pt_over_Mjj'] = sample[f'{field}_bjet_PNetRegPt'] / sample['dijet_PNetRegMass']

    # mHH variables #
    sample['pt_balance'] = sample['HH_PNetRegPt'] / (sample['lead_pt'] + sample['sublead_pt'] + sample['lead_bjet_PNetRegPt'] + sample['sublead_bjet_PNetRegPt'])

    # VH variables #
    sample['DeltaPhi_jj'] = deltaPhi(sample['nonRes_lead_bjet_phi'], sample['nonRes_sublead_bjet_phi'])
    sample['DeltaEta_jj'] = deltaEta(sample['nonRes_lead_bjet_eta'], sample['nonRes_sublead_bjet_eta'])
    isr_jet_4mom, isr_jet_bool = zh_isr_jet(sample)
    sample['isr_jet_pt'] = ak.where(isr_jet_bool, isr_jet_4mom.pt, FILL_VALUE)  # pt of isr jet
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


def main():
    sim_dir_lists = {
        # 'preEE': None,
        'postEE': None
    }
    data_dir_lists = {
        'Data': None,
    }
    
    # MC Era: total era luminosity [fb^-1] #
    luminosities = {
        'preEE': 7.9804, 
        'postEE': 26.6717
    }
    
    # Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
    #   -> Do we not need to care about other HH processes? https://arxiv.org/pdf/1910.00012.pdf
    cross_sections = {
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_HH_c
        'GluGluToHH': 34.43*0.0026,
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGG-Box-3Jets_MGG-80_13p6TeV_sherpa
        'GGJets': 88750, 
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-20to40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt20To40': 242500, 
        # https://xsdb-temp.app.cern.ch/xsdb/?columns=37748736&currentPage=0&pageSize=10&searchQuery=DAS%3DGJet_PT-40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8
        'GJetPt40': 919100, 
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#gluon_gluon_Fusion_Process
        'GluGluHToGG': 48520*0.00228,
        'GluGluHToGG_M_125': 48520*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ttH_Process
        'ttHToGG': 506.5*0.00228,
        'ttHtoGG_M_125': 506.5*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#VBF_Process
        'VBFHToGG': 3779*0.00228,
        'VBFHToGG_M_125': 3779*0.00228,
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#WH_Process + https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
        'VHToGG': (1369 + 882.4)*0.00228,
        'VHtoGG_M_125': (1369 + 882.4)*0.00228,
    }
    
    for data_era in sim_dir_lists.keys():
        all_sim_dirs_set = set(
            os.listdir(
                LPC_FILEPREFIX1+'/sim/'+data_era
            )
        )

        cross_sections_set = set([key for key in cross_sections.keys()])

        sim_dir_lists[data_era] = list(all_sim_dirs_set & cross_sections_set)
        sim_dir_lists[data_era].sort()

    for data_era in data_dir_lists.keys():
        all_data_dirs_set = set(
            os.listdir(
                LPC_FILEPREFIX+'/data/'
            )
        )

        bad_dirs_set = set()
        for data_dir in all_data_dirs_set:
            if data_dir[0] == '.':
                bad_dirs_set.add(data_dir)

        data_dir_lists[data_era] = list(all_data_dirs_set - bad_dirs_set)
        data_dir_lists[data_era].sort()
        
    for data_era, dir_list in sim_dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:
                # Load all the parquets of a single sample into an ak array
                sample = ak.concatenate(
                    [ak.from_parquet(file) for file in glob.glob(LPC_FILEPREFIX1+'/sim/'+data_era+'/'+dir_name+'/'+sample_type+'/*.parquet')]
                )
                print('loads array')

                sample['sample_name'] = dir_name

                # sample['sumGenWeights'] = sum(
                #     float(pq.read_table(file).schema.metadata['sum_genw_presel']) for file in glob.glob(
                #         LPC_FILEPREFIX+'/sim/'+data_era+'/'+dir_name+'/'+sample_type+'/*'
                #     )
                # )
                # sample['eventWeight'] = sample['genWeight'] * (luminosities[data_era] * cross_sections[dir_name] / sample['sumGenWeights'])
                sample['eventWeight'] = sample['weight_central'] * luminosities[data_era] * cross_sections[dir_name]

                add_vars(sample)
        
                destdir = LPC_FILEPREFIX+'_merged_v1/sim/'+data_era+'/'+dir_name+'/'+sample_type+'/'
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                merged_parquet = ak.to_parquet(sample, destdir+dir_name+'_merged.parquet')
                
                del sample
                print('======================== \n', destdir)

    for data_era, dir_list in data_dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:
                # Load all the parquets of a single sample into an ak array
                print(f"sample = {dir_name}")
                sample = ak.concatenate(
                    [ak.from_parquet(file) for file in glob.glob(LPC_FILEPREFIX+'/data/'+dir_name+'/'+sample_type+'/*.parquet')]
                )
                
                sample['sample_name'] = dir_name

                add_vars(sample)
        
                destdir = LPC_FILEPREFIX+'_merged_v1/data/'+dir_name+'/'+sample_type+'/'
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                merged_parquet = ak.to_parquet(sample, destdir+dir_name+'_merged.parquet')
                
                del sample
                print('======================== \n', destdir)


if __name__ == '__main__':
    main()
