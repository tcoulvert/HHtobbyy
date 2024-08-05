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
    "GGJets": r"$\gamma\gamma+3j$",
    "GJetPt20To40": r"$\gamma+j$, 20<$p_T$<40GeV",
    "GJetPt40": r"$\gamma+j$, 40GeV<$p_T$",
    "GluGluHToGG": r"ggF $H\rightarrow \gamma\gamma$",
    "VBFHToGG": r"VBF $H\rightarrow \gamma\gamma$",
    "VHToGG": r"V$H\rightarrow\gamma\gamma$",
    "ttHToGG": r"$t\bar{t}H\rightarrow\gamma\gamma$",
    # "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$",
    # "VBFHHto2B2G_CV_1_C2V_1_C3_1": r"VBF $HH\rightarrow bb\gamma\gamma$",
    # Need to fill in pretty print for BSM samples #
}
LUMINOSITIES = {
    '2022preEE': 7.9804, 
    '2022postEE': 26.6717,
    # Need to fill in lumis for other eras #
}

# Add the Data/MC agreement subplot (below histograms)
# Make condor script for the merger/processing file
# Check if the EE corrections were applied already, and if not apply them (likely in processing)

# Dictionary of variables to do MC/Data comparison
VARIABLES = {
    # key: hist.axis axes for plotting #
    # MET variables
    'puppiMET_sumEt': hist.axis.Regular(40, 20., 250, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'puppiMET $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_phi': hist.axis.Regular(20,-3.2, 3.2, name='var', label=r'puppiMET $\phi$', growth=False, underflow=False, overflow=False), 
    # jet-MET variables
    'DeltaPhi_j1MET': hist.axis.Regular(20,-3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    'DeltaPhi_j2MET': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_2,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    # jet-photon variables
    'DeltaR_jg_min': hist.axis.Regular(30, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # jet variables
    'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    'chi_t0': hist.axis.Regular(40, 0., 150, name='var', label=r'$\chi_{t0}^2$', growth=False, underflow=False, overflow=False), 
    'chi_t1': hist.axis.Regular(30, 0., 500, name='var', label=r'$\chi_{t1}^2$', growth=False, underflow=False, overflow=False), 
    # lepton variables
    'lepton1_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'lead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton2_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'sublead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton1_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead lepton $\eta$', growth=False, underflow=False, overflow=False), 
    'lepton2_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead lepton $\eta$', growth=False, underflow=False, overflow=False),
    'lepton1_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'lead lepton $\phi$', growth=False, underflow=False, overflow=False), 
    'lepton2_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'sublead lepton $\phi$', growth=False, underflow=False, overflow=False),
    # diphoton variables
    'pt': hist.axis.Regular(40, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=False, underflow=False, overflow=False), 
    'phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\gamma \gamma \phi$', growth=False, underflow=False, overflow=False),
    # angular (cos) variables
    'abs_CosThetaStar_CS': hist.axis.Regular(20, 0, 1, name='var', label=r'|cos$(\theta_{CS})$|', growth=False, underflow=False, overflow=False), 
    'abs_CosThetaStar_jj': hist.axis.Regular(20, 0, 1, name='var', label=r'|cos$(\theta_{jj})$|', growth=False, underflow=False, overflow=False), 
    # jet-lepton variables
    'leadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'leadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{sublead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{sublead})$', growth=False, underflow=False, overflow=False)
}
BLINDED_VARIABLES = {
    # dijet variables
    'dijet_mass': (
        hist.axis.Regular(50, 25., 180., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False),
        [70, 150]
    ),
    # diphoton variables
    'mass': (
        hist.axis.Regular(50, 25., 180., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False),
        [115, 135]
    )
}
# Set of extra MC variables necessary for MC/Data comparison, defined in merger.py
MC_EXTRA_VARS = {
    'luminosity', 'cross_section', 'eventWeight', 'genWeight', MC_DATA_MASK
}
DATA_EXTRA_VARS = {
    MC_DATA_MASK
}

def total_lumi(dir_lists: dict):
    total_lumi = 0
    for data_era in dir_lists.keys():
        for lumi_era in LUMINOSITIES:
            if re.search(lumi_era, data_era) is None:
                continue
            total_lumi += LUMINOSITIES[lumi_era]
    LUMINOSITIES['total_lumi'] = total_lumi

def sideband_cuts(data_era: str, sample):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (should be required in preselection, and thus be all True)
    event_mask = ak.where(sample['pt'] != FILL_VALUE, True, False) & ak.where(sample['dijet_pt'] != FILL_VALUE, True, False)
    # Require btag score above Loose WP
    EE_era_2022 = 'preEE' if re.search('preEE', data_era) is not None else 'postEE'
    event_mask = event_mask & ak.where(
        sample['lead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    ) & ak.where(
        sample['sublead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    )
    # Require at least 3 jets (to remove bbH background), extra jets coming from Ws
    event_mask = event_mask & ak.where(sample['jet3_pt'] != FILL_VALUE, True, False)
    # Require events with diphoton mass within Higgs window
    event_mask = event_mask & (
        ak.where(sample['mass'] >= 100, True, False) & ak.where(sample['mass'] <= 150, True, False)
    )
    # Mask out events with dijet mass within Higgs window
    event_mask = event_mask & (
        ak.where(sample['dijet_mass'] <= 70, True, False) | ak.where(sample['dijet_mass'] >= 150, True, False)
    )
    sample[MC_DATA_MASK] = event_mask

def get_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    for data_era in dir_lists.keys():
        if os.path.exists(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json'):
            with open(LPC_FILEPREFIX+'/'+data_era[:-7]+'/completed_samples.json', 'r') as f:
                run_samples = json.load(f)
        else:
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou first need to run the merger.py script to add the necessary variables and merge the parquets."
            )
        
        if not set(run_samples['run_samples_list']) >= set(MC_NAMES_PRETTY.keys()):
            raise Exception(
                f"Failed to find processed parquets for {data_era[:-7]}. \nYou may have run the merger.py script already, however not all of the minimal files were found."
            )
        dir_lists[data_era] = [sample_name for sample_name in MC_NAMES_PRETTY.keys()]
        for sample_name in run_samples['run_samples_list']:
            if re.search("Data", sample_name) is None:
                continue
            dir_lists[data_era].append(sample_name)

def slimmed_parquet(extra_variables: dict, sample=None):
    """
    Either slims the parquet or creates a new slim parquet.
    """
    if sample is None:
        return ak.zip(
            {field: FILL_VALUE if field != MC_DATA_MASK else False for field in set(VARIABLES.keys()) | extra_variables | set(BLINDED_VARIABLES.keys())}
        )
    else:
        return ak.zip(
            {field: sample[field] for field in set(VARIABLES.keys()) | extra_variables | set(BLINDED_VARIABLES.keys())}
        )

def make_mc_dict(dir_lists: dict):
    """
    Creates the dictionary of mc samples, where each sample is a slimmed parquet containing
      only the specified variables.
    """
    mc_dict = {}
    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            if re.search('Data', dir_name) is not None or dir_name in mc_dict:
                continue
            mc_dict[dir_name] = slimmed_parquet(MC_EXTRA_VARS)
    return mc_dict

def concatenate_records(base_sample, added_sample):
    """
    Extrapolates the ak.concatenate() functionality to copy across fields.
    """
    return ak.zip(
        {
            field: ak.concatenate((base_sample[field], added_sample[field])) for field in base_sample.fields
        }
    )

def generate_hists(MC_pqs: dict, Data_pqs: dict, variable, axis, blind_edges=None):
    # https://indico.cern.ch/event/1433936/ #
    # Generate MC hist stack
    mc_hists = {}
    for dir_name, sample in MC_pqs.items():
        # Blinds a region of the plot if necessary
        if blind_edges is not None:
            sample['MC_Data_mask'] = sample['MC_Data_mask'] & (
                (sample[variable] < blind_edges[0]) | (sample[variable] > blind_edges[1])
            )
        mc_hists[MC_NAMES_PRETTY[dir_name]] = hist.Hist(axis, storage='weight').fill(
            var=ak.where(sample['MC_Data_mask'], sample[variable], FILL_VALUE),
            weight=sample['eventWeight']
        )
    mc_stack = hist.Stack.from_dict(mc_hists)

    # Generate data hist
    data_ak = ak.zip({variable: FILL_VALUE})
    for dir_name, sample in Data_pqs.items():
        if blind_edges is not None:
            sample['MC_Data_mask'] = sample['MC_Data_mask'] & (
                (sample[variable] < blind_edges[0]) | (sample[variable] > blind_edges[1])
            )
        data_ak[variable] = ak.concatenate(
            (data_ak[variable], ak.where(sample['MC_Data_mask'], sample[variable], FILL_VALUE))
        )
    data_hist = hist.Hist(axis).fill(var=data_ak[variable])

    # Generate ratio subplot hist
    mc_ak = ak.zip({variable: FILL_VALUE})
    
    mc_weights_ak = ak.zip({variable: FILL_VALUE})
    for dir_name, sample in MC_pqs.items():
        if blind_edges is not None:
            sample['MC_Data_mask'] = sample['MC_Data_mask'] & (
                (sample[variable] < blind_edges[0]) | (sample[variable] > blind_edges[1])
            )
        mc_ak[variable] = ak.concatenate(
            (mc_ak[variable], ak.where(sample['MC_Data_mask'], sample[variable], FILL_VALUE))
        )
        mc_weights_ak[variable] = ak.concatenate(
            (mc_weights_ak[variable], sample['eventWeight'])
        )
    ratio_hist = hist.Hist(
        hist.axis.StrCategory([], growth=True, name="cat"),
        axis,
        storage='weight'
    )
    ratio_hist.fill(cat="numer", var=data_ak[variable])
    ratio_hist.fill(cat="denom", var=mc_ak[variable], weight=mc_weights_ak[variable])

    # fig,axs = plt.subplots(2, 1, sharex=True, height_ratios=[4,1])

    # axs = axs.flatten()

    # h[hist.loc("numer"),:].plot_ratio(
    #     h[hist.loc("denom"),:],
    #     rp_num_label="numer",
    #     rp_denom_label="denom",
    #     ax_dict={"main_ax":axs[0],"ratio_ax":axs[1]}
    # )
    # axs[0].set_xlabel("")
    # axs[0].set_yscale("log")
    # axs[0].set_ylim(1e-1,None)
    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0.05)

    return mc_stack, data_hist, ratio_hist

def plot(variable, mc_hist, data_hist, ratio_hist):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8)
    )
    mc_hist.plot(
        stack=True, ax=axs[0], linewidth=3, histtype="fill", sort='yield_r'
    )
    hep.histplot(
        data_hist, ax=axs[0], linewidth=3, histtype="errorbar", color="black", label=f"CMS Data"
    )
    # Make ratio subplot
    ratio_hist[hist.loc("numer"),:].plot_ratio(
        ratio_hist[hist.loc("denom"),:], 
        ax_dict={"main_ax":axs[0],"ratio_ax":axs[1]}
    )
    # plt.tight_layout()
    # fig.subplots_adjust(hspace=0.05)
    
    hep.cms.lumitext(f"{LUMINOSITIES['total_lumi']:.2f}fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Shrink current axis by 20%
    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.75])
    # Put a legend to the right of the current axis
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    # Make angular and chi^2 plots linear, otherwise log
    if re.match('chi_t', variable) is None and re.match('DeltaPhi', variable) is None:
        axs[0].set_yscale('log')
    else:
        axs[0].set_yscale('linear')
    # Save out the plot
    if not os.path.exists(DESTDIR):
        os.mkdir(DESTDIR)
    plt.savefig(f'{DESTDIR}/1dhist_{variable}_MC_Data.pdf')
    plt.savefig(f'{DESTDIR}/1dhist_{variable}_MC_Data.png')
    plt.close()
    
def main():
    """
    Performs the Data-MC comparison.
    """
    # Minimal data files for MC-Data comparison for ttH-Killer variables
    dir_lists = {
        'Run3_2022preEE_merged': None,
        'Run3_2022postEE_merged': None,
        # Need to add other data eras eventually (2023, etc)
    }
    get_dir_lists(dir_lists)
    total_lumi(dir_lists)

    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    MC_pqs = make_mc_dict(dir_lists)
    Data_pqs = {}

    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not currently computed in merger.py.
                sample = ak.concatenate(
                    [ak.from_parquet(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/'+file) for file in os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/')]
                )
                
                # perform necessary cuts to enter ttH enriched region
                sideband_cuts(data_era, sample)

                # Checks if sample is Data (True) or MC (False)
                #   -> slims parquet to only include desired variables (to save RAM, if not throttling RAM feel free to not do the slimming)
                if re.match('Data', dir_name) is not None:
                    # Data_pqs[data_era+dir_name] = slimmed_parquet(DATA_EXTRA_VARS, sample)
                    Data_pqs[dir_name] = slimmed_parquet(DATA_EXTRA_VARS, sample)
                else:
                    MC_pqs[dir_name] = concatenate_records(
                        MC_pqs[dir_name], slimmed_parquet(MC_EXTRA_VARS, sample)
                    )
                
                del sample

    # Ploting over variables for MC and Data
    for variable, axis in VARIABLES.items():
        mc_hist, data_hist, ratio_hist = generate_hists(MC_pqs, Data_pqs, variable, axis)
        plot(variable, mc_hist, data_hist, ratio_hist)

    # Ploting over variables for MC and Data
    for variable, (axis, blind_edges) in BLINDED_VARIABLES.items():
        mc_hist, data_hist, ratio_hist = generate_hists(MC_pqs, Data_pqs, variable, axis, blind_edges=blind_edges)
        plot(variable, mc_hist, data_hist, ratio_hist)

if __name__ == '__main__':
    main()