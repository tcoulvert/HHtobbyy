import glob
import json
import math
import os
import re
import sys
import subprocess

import awkward as ak
import duckdb
import pyarrow.parquet as pq
import vector as vec
vec.register_awkward()

LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v1"

def ttH_enriched_cuts(sample):
    # In Run2 they did comparison on events passing HHMVA > 0.29
    #   -> replicate using Yibo's cutbased analysis and/or BDT with the cut
    #   at the same signal efficiency as >0.29 in Run2
    #
    # OR just go to sideband region with enriched bjets, but no diHiggs
    #   -> cut on btag score for both bjets, dijet mass NOT in Higgs mass
    #   window (<70Gev or >150Gev, check values based on HHbbgg presentations),
    #   don't cut on diphoton b/c thats in the ttH background as well, focus on making bjet enriched?
    pass

def main():
    dir_lists = {
        'Run3_2022preEE_merged': None,
        'Run3_2022postEE_merged': None
    }

    # Dictionary of variables to do MC/Data comparison
    variables = {
        'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
        'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
        'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
        'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
        'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
        'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
        'abs_CosThetaStar_CS', 'abs_CosThetaStar_jj', # angular variables
        'dijet_mass', # mass of b-dijet (resonance for H->bb)
        'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
        'subleadBjet_leadLepton', 'subleadBjet_subleadLepton'
    }
    
    for data_era in dir_lists.keys():
        if os.path.exists(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json'):
            with open(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json', 'r') as f:
                run_samples = json.load(f)
        else:
            raise Exception(f"Failed to find processed parquest for {data_era[:-7]}. \nYou first need to run the merger.py script to add the necessary variables and merge the parquets.")
        dir_lists[data_era] = run_samples['run_samples_list']

    MC_pqs = {}
    Data_pqs = {}
    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not currently computed in merger.py.
                sample = ak.concatenate(
                    [ak.from_parquet(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/'+file) for file in os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/')]
                )
                
                # perform necessary cuts to enter ttH enriched region
                ttH_enriched_cuts(sample)

                # slim parquet to only include desired variables (to save RAM, if not throttling RAM feel free to not do the slimming)
                slimmed_sample = ak.zip(
                    {
                        field: sample[field] for field in variables
                    }
                )
                if re.match('Data', dir_name) is None:  # Checks if sample is MC (True) or Data (False)
                    MC_pqs[dir_name] = ak.copy(slimmed_sample)
                else:
                    Data_pqs[dir_name] = ak.copy(slimmed_sample)
        
                
                del sample, slimmed_sample
                print('======================== \n', dir_name)

    #
    # Now do printing over variables for MC and Data
    # 
    for variable in variables:
        pass


if __name__ == '__main__':
    main()