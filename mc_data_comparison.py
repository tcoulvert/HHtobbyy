import glob
import json
import math
import os
import re
import sys
import subprocess

import awkward as ak
import duckdb
import hist
import numpy as np
import pyarrow.parquet as pq
import vector as vec
vec.register_awkward()

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})

LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v1"
SINGLE_B_WPS = {
    'preEE': {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961},
    'postEE': {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664}
}

def ttH_enriched_cuts(data_era, sample):
    # In Run2 they did comparison on events passing HHMVA > 0.29
    #   -> replicate using Yibo's cutbased analysis and/or BDT with the cut
    #   at the same signal efficiency as >0.29 in Run2
    #
    # OR just go to sideband region with enriched bjets, but no diHiggs
    #   -> cut on btag score for both bjets, dijet mass NOT in Higgs mass
    #   window (<70Gev or >150Gev, check values based on HHbbgg presentations),
    #   don't cut on diphoton b/c thats in the ttH background as well, focus on making bjet enriched?

    # Require diphoton and dijet exist (should be required in preselection, and thus be all True)
    event_mask = ak.where(sample['pt'] != -999) & ak.where(sample['dijet_pt'] != -999)

    # Require btag score above loose WP
    EEera_2022 = 'preEE' if re.search('preEE', data_era) is not None else 'postEE'
    event_mask = event_mask & ak.where(
        sample['lead_bjet_btagPNetB'] > SINGLE_B_WPS[EEera_2022]['L']
    ) & ak.where(
        sample['sublead_bjet_btagPNetB'] > SINGLE_B_WPS[EEera_2022]['L']
    )

    # Require at least 3 jets (to remove bbH background), extra jets coming from Ws
    event_mask = event_mask & ak.where(sample['jet3_pt'] != -999)

    # Mask out events with dijet mass within Higgs window
    event_mask = event_mask & (
        ak.where(sample['dijet_mass'] <= 70) | ak.where(sample['dijet_mass'] >= 150)
    )

    mask_name = 'MC_Data_mask'
    sample[mask_name] = event_mask
    return mask_name
    
def main(minimal=True):
    # Minimal data files for MC-Data comparison for ttH-Killer variables
    dir_lists = {
        'Run3_2022preEE_merged': ['Data_EraC', 'Data_EraD', 'ttHToGG'],
        'Run3_2022postEE_merged': ['Data_EraE', 'Data_EraF', 'Data_EraG', 'ttHToGG']
    }

    for data_era in dir_lists.keys():
        if os.path.exists(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json'):
            with open(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json', 'r') as f:
                run_samples = json.load(f)
        else:
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou first need to run the merger.py script to add the necessary variables and merge the parquets."
            )
        
        if not set(run_samples['run_samples_list']) >= dir_lists[data_era]:
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou may have run the merger.py script already, however not all of the minimal files were found. \nminimal files:\n{minimal_set}"
            )
        if not minimal:
            dir_lists[data_era] = run_samples['run_samples_list']

    # Dictionary of variables to do MC/Data comparison
    variables = {
        # key: hist.axis axes for plotting #
        # MET variables
        'puppiMET_sumEt': hist.axis.Regular(50, 20., 250, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=True), 
        'puppiMET_pt': hist.axis.Regular(50, 20., 250, name='var', label=r'puppiMET $p_T$ [GeV]', growth=True), 
        'puppiMET_phi': hist.axis.Regular(25,-3.2, 3.2, name='var', label=r'puppiMET $\phi$', growth=True), 
        # jet-MET variables
        'DeltaPhi_j1MET': hist.axis.Regular(50,-3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,E_T^{miss})$', growth=True), 
        'DeltaPhi_j2MET': hist.axis.Regular(50, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_2,E_T^{miss})$', growth=True), 
        # jet-photon variables
        'DeltaR_jg_min': hist.axis.Regular(50, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=True), 
        # jet variables
        'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=True), 
        'chi_t0': hist.axis.Regular(70, 0., 150, name='var', label=r'$\chi_{t0}^2$', growth=True), 
        'chi_t1': hist.axis.Regular(70, 0., 500, name='var', label=r'$\chi_{t1}^2$', growth=True), 
        # lepton variables
        'lepton1_pt': hist.axis.Regular(50, 0., 200, name='var', label=r'lead lepton $p_T$ [GeV]', growth=True), 
        'lepton2_pt': hist.axis.Regular(50, 0., 200, name='var', label=r'sublead lepton $p_T$ [GeV]', growth=True), 
        'lepton1_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead lepton $\eta$', growth=True), 
        'lepton2_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead lepton $\eta$', growth=True),
        'lepton1_phi': hist.axis.Regular(30, -3.2, 3.2, name='var', label=r'lead lepton $\phi$', growth=True), 
        'lepton2_phi': hist.axis.Regular(30, -3.2, 3.2, name='var', label=r'sublead lepton $\phi$', growth=True),
        # diphoton variables
        'pt': hist.axis.Regular(50, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=True),
        'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=True), 
        'phi': hist.axis.Regular(16, -3.2, 3.2, name='var', label=r'$\gamma \gamma \phi$', growth=True),
        # angular (cos) variables
        'abs_CosThetaStar_CS': hist.axis.Regular(25, 0, 1, name='var', label=r'|cos$(\theta_{CS})$|', growth=True), 
        'abs_CosThetaStar_jj': hist.axis.Regular(25, 0, 1, name='var', label=r'|cos$(\theta_{jj})$|', growth=True), 
        # dijet variables
        # 'dijet_mass': hist.axis.Regular(50, 25., 180., name='var', label=r'$M_{jj}$ [GeV]', growth=True), # mass of b-dijet (resonance for H->bb)
        # jet-lepton variables
        'leadBjet_leadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=True), 
        'leadBjet_subleadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{sublead})$', growth=True), 
        'subleadBjet_leadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{lead})$', growth=True), 
        'subleadBjet_subleadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{sublead})$', growth=True)
    }
    # Set of extra variables necessary for MC/Data comparison
    extra_variables = {
        'luminosity', 'cross_section', 'eventWeight'
    }

    MC_pqs = {}
    Data_pqs = {}
    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not currently computed in merger.py.
                sample = ak.concatenate(
                    [ak.from_parquet(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/'+file) for file in os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/')]
                )
                
                # perform necessary cuts to enter ttH enriched region
                extra_variables.add(ttH_enriched_cuts(data_era, sample))

                # slim parquet to only include desired variables (to save RAM, if not throttling RAM feel free to not do the slimming)
                slimmed_sample = ak.zip(
                    {
                        field: sample[field] for field in (set(variables.keys()) + extra_variables)
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