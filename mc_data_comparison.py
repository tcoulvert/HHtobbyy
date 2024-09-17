import glob
import json
import os
import re
import warnings

import awkward as ak
import hist
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

# LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v1"
# LPC_FILEPREFIX = "/uscms/home/tsievert/nobackup/XHYbbgg/HiggsDNA_official/output_test_HH"
LPC_FILEPREFIX = "/uscms/home/tsievert/nobackup/XHYbbgg/HiggsDNA_official/output_test_ttH_2"
DESTDIR = 'v1_comparison_plots_test_ttH_unweighted'
APPLY_WEIGHTS = False
SINGLE_B_WPS = {
    'preEE': {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961},
    'postEE': {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664}
}
MC_DATA_MASK = 'MC_Data_mask'
FILL_VALUE = -999
MC_NAMES_PRETTY = {
    # "GGJets": r"$\gamma\gamma+3j$",
    # "GJetPt20To40": r"$\gamma+j$, 20<$p_T$<40GeV",
    # "GJetPt40": r"$\gamma+j$, 40GeV<$p_T$",
    # "GluGluHToGG": r"ggF $H\rightarrow \gamma\gamma$",
    # "VBFHToGG": r"VBF $H\rightarrow \gamma\gamma$",
    # "VHToGG": r"V$H\rightarrow\gamma\gamma$",
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
    'jet1_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'jet2_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'sublead jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
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
    # single photon variables
    'lead_pt': hist.axis.Regular(40, 20., 200, name='var', label=r' lead $\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False), 
    'sublead_pt': hist.axis.Regular(40, 20., 200, name='var', label=r' sublead $\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    # diphoton variables
    'pt': hist.axis.Regular(40, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=False, underflow=False, overflow=False), 
    'phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\gamma \gamma \phi$', growth=False, underflow=False, overflow=False),
    # angular (cos) variables
    'CosThetaStar_CS': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{CS})$', growth=False, underflow=False, overflow=False), 
    'CosThetaStar_jj': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{jj})$', growth=False, underflow=False, overflow=False), 
    # jet-lepton variables
    'leadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'leadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{sublead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{sublead})$', growth=False, underflow=False, overflow=False),
    # Electron variables
    'lead_electron_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'lead electron $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lead_electron_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead electron $\eta$', growth=False, underflow=False, overflow=False), 
    'lead_electron_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'lead electron $\phi$', growth=False, underflow=False, overflow=False), 
    'sublead_electron_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'sublead electron $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'sublead_electron_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead electron $\eta$', growth=False, underflow=False, overflow=False),
    'sublead_electron_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'sublead electron $\phi$', growth=False, underflow=False, overflow=False),
    'lead_electron_MVA': hist.axis.Regular(30, 0.8, 1., name='var', label=r'lead electron mvaIso', growth=False, underflow=False, overflow=False), 
    'sublead_electron_MVA': hist.axis.Regular(30, 0.8, 1., name='var', label=r'sublead electron mvaIso', growth=False, underflow=False, overflow=False),
    # Muon variables
    'lead_muon_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'lead muon $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lead_muon_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead muon $\eta$', growth=False, underflow=False, overflow=False), 
    'lead_muon_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'lead muon $\phi$', growth=False, underflow=False, overflow=False), 
    'sublead_muon_pt': hist.axis.Regular(40, 0., 200, name='var', label=r'sublead muon $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'sublead_muon_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead muon $\eta$', growth=False, underflow=False, overflow=False),
    'sublead_muon_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'sublead muon $\phi$', growth=False, underflow=False, overflow=False),
    'lead_muon_MVA': hist.axis.Regular(30, 0., 1., name='var', label=r'lead muon mvaMuID', growth=False, underflow=False, overflow=False), 
    'sublead_muon_MVA': hist.axis.Regular(30, 0., 1., name='var', label=r'sublead muon mvaMuID', growth=False, underflow=False, overflow=False),
}
BLINDED_VARIABLES = {
    # dijet variables
    'dijet_mass': (
        hist.axis.Regular(24, 70., 190., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False),
        [100, 150]
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
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = ak.where(sample['pt'] != FILL_VALUE, True, False) & ak.where(sample['dijet_pt'] != FILL_VALUE, True, False)
    # # Require btag score above Loose WP
    # EE_era_2022 = 'preEE' if re.search('preEE', data_era) is not None else 'postEE'
    # event_mask = event_mask & ak.where(
    #     sample['lead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    # ) & ak.where(
    #     sample['sublead_bjet_btagPNetB'] > SINGLE_B_WPS[EE_era_2022]['L'], True, False
    # )
    # # Require at least 3 jets (to remove bbH background), extra jets coming from Ws
    # event_mask = event_mask & ak.where(sample['jet3_pt'] != FILL_VALUE, True, False)
    # # Require events with diphoton mass within Higgs window
    # event_mask = event_mask & (
    #     ak.where(sample['mass'] >= 100, True, False) & ak.where(sample['mass'] <= 150, True, False)
    # )
    # # Mask out events with dijet mass within Higgs window
    # event_mask = event_mask & (
    #     ak.where(sample['dijet_mass'] <= 100, True, False) | ak.where(sample['dijet_mass'] >= 150, True, False)
    # )
    sample[MC_DATA_MASK] = event_mask

def get_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    for data_era in dir_lists.keys():
        # if os.path.exists(LPC_FILEPREFIX+'/'+data_era[:-10]+'/completed_samples.json'): # -7 -> -10
        #     with open(LPC_FILEPREFIX+'/'+data_era[:-10]+'/completed_samples.json', 'r') as f:
        #         processed_samples = json.load(f)
        # else:
        #     raise Exception(
        #         f"Failed to find processed parquets for {data_era[:-7]}. \nYou first need to run the merger.py script to add the necessary variables and merge the parquets."
        #     )
        processed_samples = [
            sample[sample.rfind('/')+1:] for sample in glob.glob(LPC_FILEPREFIX+'/'+data_era+'/*')
        ]
        
        if not set(processed_samples) >= set(MC_NAMES_PRETTY.keys()):
            raise Exception(
                f"Failed to find processed parquets for {data_era}. \nYou may have run the merger.py script already, however not all of the minimal files were found."
            )
        dir_lists[data_era] = [sample_name for sample_name in MC_NAMES_PRETTY.keys()]
        for sample_name in processed_samples:
            if re.search("Data", sample_name) is None or sample_name not in set(os.listdir(LPC_FILEPREFIX+'/'+data_era)):
                continue
            dir_lists[data_era].append(sample_name)

def slimmed_parquet(extra_variables: dict, sample=None):
    """
    Either slims the parquet or creates a new slim parquet.
    """
    if sample is None:
        return ak.zip(
            {field: FILL_VALUE if field != MC_DATA_MASK and field != 'eventWeight' else False for field in set(VARIABLES.keys()) | extra_variables | set(BLINDED_VARIABLES.keys())}
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

def generate_hists(MC_pqs: dict, Data_pqs: dict, variable: str, axis, blind_edges=None):
    # https://indico.cern.ch/event/1433936/ #
    # Generate MC hist stack
    mc_hists = {}
    for dir_name, sample in MC_pqs.items():
        # Blinds a region of the plot if necessary
        if blind_edges is not None:
            mask = (
                (sample[variable] < blind_edges[0]) | (sample[variable] > blind_edges[1])
            )
        else:
            mask = sample['MC_Data_mask']
        if APPLY_WEIGHTS:
            mc_hists[MC_NAMES_PRETTY[dir_name]] = hist.Hist(axis, storage='weight').fill(
                var=ak.where(mask, sample[variable], FILL_VALUE),
                weight=sample['eventWeight']
            )
        else:
            mc_hists[MC_NAMES_PRETTY[dir_name]] = hist.Hist(axis).fill(
                var=ak.where(mask, sample[variable], FILL_VALUE)
            )

    # Generate data hist
    data_ak = ak.zip({variable: FILL_VALUE})
    for sample in Data_pqs.values():
        if blind_edges is not None:
            mask = (
                (sample[variable] < blind_edges[0]) | (sample[variable] > blind_edges[1])
            )
        else:
            mask = sample['MC_Data_mask']
        data_ak[variable] = ak.concatenate(
            (data_ak[variable], ak.where(mask, sample[variable], FILL_VALUE))
        )
    data_hist = hist.Hist(axis).fill(var=data_ak[variable])

    # Generate ratio dict
    ratio_dict = {
        'mc_values': np.array([0.0 for _ in range(axis.size)]),
        'w2': np.array([0.0 for _ in range(axis.size)])
    }
    for mc_hist in mc_hists.values():
        ratio_dict['mc_values'] += mc_hist.values().flatten()
        ratio_dict['w2'] += mc_hist.variances().flatten()
    ratio_dict['data_values'] = data_hist.values().flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio_dict['ratio'] = ratio_dict['data_values'] / ratio_dict['mc_values']

    return mc_hists, data_hist, ratio_dict

def plot_ratio(ratio, mpl_ax, hist_axis, numer_err=None, denom_err=None, central_value=1.0):
    """
    Does the ratio plot (code copied from Hist.plot_ratio_array b/c they don't
      do what we need.)
    """
    # Set 0 and inf to nan to hide during plotting
    for arr in [ratio, numer_err, denom_err]:
        if arr is None:
            continue
        arr[arr == 0] = np.nan
        arr[np.isinf(arr)] = np.nan

    mpl_ax.set_ylim(0, 2)
    mpl_ax.axhline(
        central_value, color="black", linestyle="dashed", linewidth=1.0
    )
    mpl_ax.errorbar(
        hist_axis.centers[0], ratio, yerr=numer_err,
        color="black", marker="o", linestyle="none"
    )

    if denom_err is not None:
        mpl_ax.bar(
            hist_axis.centers[0], height=denom_err * 2, width=(hist_axis.centers[0][1] - hist_axis.centers[0][0]), 
            bottom=(central_value - denom_err), color="green", alpha=0.5, hatch='//'
        )

def ratio_error(numer_values, denom_values, numer_err, denom_err):
    ratio_err =  np.sqrt(
        np.power(denom_values, -2) * (
            np.power(numer_err, 2) + (
                np.power(numer_values / denom_values, 2) * np.power(denom_err, 2)
            )
        )
    )
    return ratio_err

def plot(variable: str, mc_hist: dict, data_hist: hist.Hist, ratio_dict: dict):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # print(f"{variable} integral = {np.sum(data_hist.values(), axis=0)}")
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8)
    )
    hep.histplot(
        list(mc_hist.values()), label=list(mc_hist.keys()), 
        # w2=np.vstack((np.tile(np.zeros_like(ratio_dict['w2']), (len(mc_hist)-1, 1)), ratio_dict['w2'])),
        # yerr=ratio_dict['w2'],
        stack=True, ax=axs[0], linewidth=3, histtype="fill", sort="yield"
    )
    # hep.histplot(
    #     data_hist, ax=axs[0], linewidth=3, histtype="errorbar", color="black", label=f"CMS Data"
    # )
    # Calculates the numer(denom) ratio error as: numer(denom)_hist_error / numer(denom)_hist_value
    #   -> suppresses warning coming from 0, inf, and NaN divides
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # numer_err = np.sqrt(ratio_dict['data_values']) * ratio_dict['ratio'] / ratio_dict['data_values']
        # denom_err = np.sqrt(ratio_dict['w2']) / ratio_dict['mc_values']
        ratio_err = ratio_error(ratio_dict['data_values'], ratio_dict['mc_values'], np.sqrt(ratio_dict['data_values']), np.sqrt(ratio_dict['w2']))
    # plot_ratio(
    #     ratio_dict['ratio'], axs[1], 
    #     numer_err=ratio_err,
    #     denom_err=None,
    #     hist_axis=data_hist.axes
    # )
    
    # Plotting niceties #
    hep.cms.lumitext(f"{LUMINOSITIES['total_lumi']:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), reverse=True)
    plt.tight_layout(rect=(0, 0.01, 1, 1))
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(data_hist.axes.label[0])
    # Make angular and chi^2 plots linear, otherwise log
    if re.search('chi_t', variable) is None and re.search('DeltaPhi', variable) is None and re.search('mass', variable) is None:
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
        # 'Run3_2022preEE_merged_v2': None,
        'Run3_2022postEE_merged_v2': None,
        # Need to add other data eras eventually (2023, etc)
    }
    get_dir_lists(dir_lists)
    total_lumi(dir_lists)

    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    MC_pqs = make_mc_dict(dir_lists)
    Data_pqs = {}

    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not computed in merger.py.
                sample = ak.concatenate(
                    [ak.from_parquet(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/'+file) for file in os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/')]
                )
                print(f"num events: {ak.num(sample['jet1_pt'], axis=0)}")
                
                # perform necessary cuts to enter ttH enriched region
                sideband_cuts(data_era, sample)

                # Checks if sample is Data (True) or MC (False)
                #   -> slims parquet to only include desired variables (to save RAM, if not throttling RAM feel free to not do the slimming)
                if re.match('Data', dir_name) is not None:
                    Data_pqs[data_era+dir_name] = slimmed_parquet(DATA_EXTRA_VARS, sample)
                else:
                    MC_pqs[dir_name] = concatenate_records(
                        MC_pqs[dir_name], slimmed_parquet(MC_EXTRA_VARS, sample)
                    )
                
                del sample
                print('======================== \n', dir_name)

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