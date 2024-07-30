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
DESTDIR = 'v1_comparison_plots'
SINGLE_B_WPS = {
    'preEE': {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961},
    'postEE': {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664}
}
MC_DATA_MASK = 'MC_Data_mask'
FILL_VALUE = -999
MC_NAMES_PRETTY = {
    'ttHToGG': r"$t\bar{t}H\rightarrow\gamma\gamma$",
    "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$"
}

# Dictionary of variables to do MC/Data comparison
VARIABLES = {
    # key: hist.axis axes for plotting #
    # MET variables
    'puppiMET_sumEt': hist.axis.Regular(50, 20., 250, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_pt': hist.axis.Regular(50, 20., 250, name='var', label=r'puppiMET $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_phi': hist.axis.Regular(25,-3.2, 3.2, name='var', label=r'puppiMET $\phi$', growth=False, underflow=False, overflow=False), 
    # jet-MET variables
    'DeltaPhi_j1MET': hist.axis.Regular(50,-3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    'DeltaPhi_j2MET': hist.axis.Regular(50, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_2,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    # jet-photon variables
    'DeltaR_jg_min': hist.axis.Regular(50, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # jet variables
    'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    'chi_t0': hist.axis.Regular(70, 0., 150, name='var', label=r'$\chi_{t0}^2$', growth=False, underflow=False, overflow=False), 
    'chi_t1': hist.axis.Regular(70, 0., 500, name='var', label=r'$\chi_{t1}^2$', growth=False, underflow=False, overflow=False), 
    # lepton variables
    'lepton1_pt': hist.axis.Regular(50, 0., 200, name='var', label=r'lead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton2_pt': hist.axis.Regular(50, 0., 200, name='var', label=r'sublead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton1_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead lepton $\eta$', growth=False, underflow=False, overflow=False), 
    'lepton2_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead lepton $\eta$', growth=False, underflow=False, overflow=False),
    'lepton1_phi': hist.axis.Regular(30, -3.2, 3.2, name='var', label=r'lead lepton $\phi$', growth=False, underflow=False, overflow=False), 
    'lepton2_phi': hist.axis.Regular(30, -3.2, 3.2, name='var', label=r'sublead lepton $\phi$', growth=False, underflow=False, overflow=False),
    # diphoton variables
    'pt': hist.axis.Regular(50, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=False, underflow=False, overflow=False), 
    'phi': hist.axis.Regular(16, -3.2, 3.2, name='var', label=r'$\gamma \gamma \phi$', growth=False, underflow=False, overflow=False),
    # angular (cos) variables
    'abs_CosThetaStar_CS': hist.axis.Regular(25, 0, 1, name='var', label=r'|cos$(\theta_{CS})$|', growth=False, underflow=False, overflow=False), 
    'abs_CosThetaStar_jj': hist.axis.Regular(25, 0, 1, name='var', label=r'|cos$(\theta_{jj})$|', growth=False, underflow=False, overflow=False), 
    # dijet variables
    # 'dijet_mass': hist.axis.Regular(50, 25., 180., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False), # mass of b-dijet (resonance for H->bb)
    # jet-lepton variables
    'leadBjet_leadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'leadBjet_subleadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{sublead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_leadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_subleadLepton': hist.axis.Regular(50, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{sublead})$', growth=False, underflow=False, overflow=False)
}
# Set of extra MC variables necessary for MC/Data comparison, defined in merger.py
MC_EXTRA_VARS = {
    'luminosity', 'cross_section', 'eventWeight', 'genWeight', MC_DATA_MASK
}
DATA_EXTRA_VARS = {
    MC_DATA_MASK
}

def ttH_enriched_cuts(data_era: str, sample):
    # In Run2 they did comparison on events passing HHMVA > 0.29
    #   -> replicate using Yibo's cutbased analysis and/or BDT with the cut
    #   at the same signal efficiency as >0.29 in Run2
    #
    # OR just go to sideband region with enriched bjets, but no diHiggs
    #   -> cut on btag score for both bjets, dijet mass NOT in Higgs mass
    #   window (<70Gev or >150Gev, check values based on HHbbgg presentations),
    #   don't cut on diphoton b/c thats in the ttH background as well, focus on making bjet enriched?

    # Require diphoton and dijet exist (should be required in preselection, and thus be all True)
    event_mask = ak.where(sample['pt'] != FILL_VALUE, True, False) & ak.where(sample['dijet_pt'] != FILL_VALUE, True, False)

    # # Require btag score above Loose WP
    # EE_era_2022 = 'preEE' if re.search('preEE', data_era) is not None else 'postEE'
    # event_mask = event_mask & ak.where(
    #     sample['lead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    # ) & ak.where(
    #     sample['sublead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    # )

    # Require at least 3 jets (to remove bbH background), extra jets coming from Ws
    # event_mask = event_mask & ak.where(sample['jet3_pt'] != FILL_VALUE, True, False)

    # # Require events with diphoton mass within Higgs window
    # event_mask = event_mask & (
    #     ak.where(sample['mass'] >= 100, True, False) & ak.where(sample['mass'] <= 150, True, False)
    # )

    # # Mask out events with dijet mass within Higgs window
    # event_mask = event_mask & (
    #     ak.where(sample['dijet_mass'] <= 70, True, False) | ak.where(sample['dijet_mass'] >= 150, True, False)
    # )

    # Mask out events with diphoton mass within Higgs window
    event_mask = event_mask & (
        ak.where(sample['mass'] <= 100, True, False) & ak.where(sample['mass'] >= 150, True, False)
    )

    sample[MC_DATA_MASK] = event_mask

def get_dir_lists(dir_lists: dict, minimal: bool):
    for data_era in dir_lists.keys():
        if os.path.exists(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json'):
            with open(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json', 'r') as f:
                run_samples = json.load(f)
        else:
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou first need to run the merger.py script to add the necessary variables and merge the parquets."
            )
        
        if not set(run_samples['run_samples_list']) >= set(dir_lists[data_era]):
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou may have run the merger.py script already, however not all of the minimal files were found."
            )
        if not minimal:
            dir_lists[data_era] = run_samples['run_samples_list']

def slimmed_parquet(extra_variables: dict, sample=None):
    if sample is None:
        return ak.zip(
            {field: FILL_VALUE for field in set(VARIABLES.keys()) | extra_variables}
        )
    else:
        return ak.zip(
            {field: sample[field] for field in set(VARIABLES.keys()) | extra_variables}
        )

def make_mc_dict(dir_lists: dict):
    mc_dict = {}
    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            if re.search('Data', dir_name) is not None or dir_name in mc_dict:
                continue
            mc_dict[dir_name] = slimmed_parquet(MC_EXTRA_VARS)
    return mc_dict

def concatenate_records(base_sample, added_sample):
    return ak.zip(
        {
            field: ak.concatenate((base_sample[field], added_sample[field])) for field in base_sample.fields
        }
    )
    
def main(minimal=True):
    # Minimal data files for MC-Data comparison for ttH-Killer variables
    dir_lists = {
        'Run3_2022preEE_merged': ['Data_EraC', 'Data_EraD', 'ttHToGG'],
        'Run3_2022postEE_merged': ['Data_EraE', 'Data_EraF', 'Data_EraG', 'ttHToGG']
    }
    get_dir_lists(dir_lists, minimal)

    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    MC_pqs = make_mc_dict(dir_lists)
    Data_pqs = {"Data": slimmed_parquet(DATA_EXTRA_VARS)}

    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not currently computed in merger.py.
                sample = ak.concatenate(
                    [ak.from_parquet(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/'+file) for file in os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/')]
                )
                
                # perform necessary cuts to enter ttH enriched region
                ttH_enriched_cuts(data_era, sample)

                # Checks if sample is Data (True) or MC (False)
                #   -> slims parquet to only include desired variables (to save RAM, if not throttling RAM feel free to not do the slimming)
                if re.match('Data', dir_name) is not None:
                    Data_pqs["Data"] = concatenate_records(
                        Data_pqs["Data"], slimmed_parquet(DATA_EXTRA_VARS, sample)
                    )
                else:
                    MC_pqs[dir_name] = concatenate_records(
                        MC_pqs[dir_name], slimmed_parquet(MC_EXTRA_VARS, sample)
                    )
                
                del sample
                # print('======================== \n', dir_name)

    # Now do printing over variables for MC and Data
    for variable, axis in VARIABLES.items():
        # Initiate figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate MC hist stack
        mc_hists = {}
        for dir_name, sample in MC_pqs.items():
            # print(f"{dir_name}: \n{sample['genWeight']}")  # +/- 3.1
            mc_hists[dir_name] = hist.Hist(axis, storage='weight', label=MC_NAMES_PRETTY[dir_name]).fill(
                var=ak.where(sample['MC_Data_mask'], sample[variable], FILL_VALUE),
                weight=sample['eventWeight']
                # weight=ak.where(sample['genWeight'] < 0, -1, 1) * sample['luminosity'] * sample['cross_section']
            )
        mc_stack = hist.Stack.from_dict(mc_hists)
        mc_stack.plot(
            # stack=True, ax=ax, linewidth=3, histtype="fill", color=mc_colors, label=mc_labels, color=[cm(i/len(MC_pqs)) for i in range(len(MC_pqs))]
            stack=True, ax=ax, linewidth=3, histtype="fill"
        )

        # Generate data hist
        data_ak = ak.zip({variable: FILL_VALUE})
        for dir_name, sample in Data_pqs.items():
            data_ak[variable] = ak.concatenate(
                (data_ak[variable], ak.where(sample['MC_Data_mask'], sample[variable], FILL_VALUE))
            )
        data_stack = hist.Hist(axis).fill(var=data_ak[variable])
        hep.histplot(
            data_stack, ax=ax, linewidth=3, histtype="errorbar", color="black", label=f"CMS Data"
        )

        # Plotting niceties
        hep.cms.lumitext(f"{2022} (13.6 TeV)", ax=ax)
        hep.cms.text("Work in Progress", ax=ax)
        ax.legend(ncol=1, loc = 'best')
        ax.set_yscale('log')
        # if re.match('chi_t', variable) is None and re.match('DeltaPhi', variable) is None:
        #     ax.set_yscale('log')
        # else:
        #     ax.set_yscale('linear')
        if not os.path.exists(DESTDIR):
            os.mkdir(DESTDIR)
        plt.savefig(f'{DESTDIR}/1dhist_{variable}_MC_Data.pdf')
        plt.savefig(f'{DESTDIR}/1dhist_{variable}_MC_Data.png')
        plt.close()


if __name__ == '__main__':
    main(False)