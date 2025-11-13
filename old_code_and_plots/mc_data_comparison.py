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
# LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/Run3_2022_merged_v1"
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/"
Run3_2022 = 'Run3_2022_merged/sim'
Run3_2023 = 'Run3_2023_merged/sim'
LPC_FILEPREFIX_SIM = LPC_FILEPREFIX +'/sim'
LPC_FILEPREFIX_DATA = LPC_FILEPREFIX +'/data'
DESTDIR = 'v2_comparison_plots'
if not os.path.exists(DESTDIR):
    os.makedirs(DESTDIR)

APPLY_WEIGHTS = True
SINGLE_B_WPS = {
    'preEE': {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961},
    'postEE': {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664}
}
MC_DATA_MASK = 'MC_Data_mask'
FILL_VALUE = -999
MC_NAMES_PRETTY = {
    # non-resonant
    "GGJets": r"$\gamma\gamma+3j$",
    "GJetPt20To40": r"$\gamma+j$, 20<$p_T$<40GeV",
    "GJetPt40": r"$\gamma+j$, 40GeV<$p_T$",
    # single-H
    "GluGluHToGG": r"ggF $H\rightarrow \gamma\gamma$",
    'GluGluHToGG_M_125': r"ggF $H\rightarrow \gamma\gamma$",
    "VBFHToGG": r"VBF $H\rightarrow \gamma\gamma$",
    'VBFHToGG_M_125': r"VBF $H\rightarrow \gamma\gamma$",
    "VHToGG": r"V$H\rightarrow\gamma\gamma$",
    'VHtoGG_M_125': r"V$H\rightarrow\gamma\gamma$",
    "ttHToGG": r"$t\bar{t}H\rightarrow\gamma\gamma$",
    'ttHtoGG_M_125': r"$t\bar{t}H\rightarrow\gamma\gamma$",
    'BBHto2G_M_125': r"$b\bar{b}H\rightarrow\gamma\gamma$",
    # signal
    "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$",
    'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': r"ggF $HH\rightarrow bb\gamma\gamma$",
}
LUMINOSITIES = {
    '2022preEE': 7.9804, 
    '2022postEE': 26.6717,
    # Need to fill in lumis for other eras #
}
LUMINOSITIES['total_lumi'] = sum(LUMINOSITIES.values())

# Dictionary of variables
VARIABLES = {
    # key: hist.axis axes for plotting #
    # MET variables
    'puppiMET_sumEt': hist.axis.Regular(40, 150., 2000, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    # jet-photon variables
    'nonRes_DeltaR_jg_min': hist.axis.Regular(30, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # jet variables
    'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    # ATLAS variables #
    'RegPt_balance': hist.axis.Regular(100, 0., 2., name='var', label=r'$HH p_{T} / (\gamma1 p_{T} + \gamma2 p_{T} + j1 p_{T} + j2 p_{T})$', growth=False, underflow=False, overflow=False), 
    # photon variables
    'lead_mvaID_run3': hist.axis.Regular(100, -1., 1, name='var', label=r'lead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False), 
    'sublead_mvaID_run3': hist.axis.Regular(100, -1., 1, name='var', label=r'sublead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False), 
}
BLINDED_VARIABLES = {
    # dijet variables
    'dijet_PNetRegMass': (
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
    'eventWeight', 'weight', MC_DATA_MASK
}
DATA_EXTRA_VARS = {
    MC_DATA_MASK
}

def sideband_cuts(sample):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = (
        sample['nonRes_has_two_btagged_jets'] 
        & sample['is_nonRes']
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        )
    )
    sample[MC_DATA_MASK] = event_mask

def get_mc_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    
    for data_era in dir_lists.keys():
        processed_samples = [
            sample[sample.rfind('/')+1:] for sample in glob.glob(LPC_FILEPREFIX_SIM+'/'+data_era+'/*')
        ]
        
        # if not set(processed_samples) >= set(MC_NAMES_PRETTY.keys()):
        #     raise Exception(
        #         f"Failed to find processed parquets for {data_era}. \nYou may have run the merger.py script already, however not all of the minimal files were found."
        #     )
        common_samples = list(set(MC_NAMES_PRETTY.keys()) & set(processed_samples))
        common_samples.sort()
        dir_lists[data_era] = [sample_name for sample_name in common_samples]

def get_data_dir_lists(dir_lists: dict):
    
    for data_era in dir_lists.keys():
        processed_samples = [
            sample[sample.rfind('/')+1:] for sample in glob.glob(LPC_FILEPREFIX_DATA+'/*')
        ]
        processed_samples.sort()
        
        dir_lists[data_era] = processed_samples

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
        w2=np.vstack((np.tile(np.zeros_like(ratio_dict['w2']), (len(mc_hist)-1, 1)), ratio_dict['w2'])),
        # yerr=ratio_dict['w2'],
        stack=True, ax=axs[0], linewidth=3, histtype="fill", sort="yield"
    )
    hep.histplot(
        data_hist, ax=axs[0], linewidth=3, histtype="errorbar", color="black", label=f"CMS Data"
    )
    # Calculates the numer(denom) ratio error as: numer(denom)_hist_error / numer(denom)_hist_value
    #   -> suppresses warning coming from 0, inf, and NaN divides
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # numer_err = np.sqrt(ratio_dict['data_values']) * ratio_dict['ratio'] / ratio_dict['data_values']
        # denom_err = np.sqrt(ratio_dict['w2']) / ratio_dict['mc_values']
        ratio_err = ratio_error(ratio_dict['data_values'], ratio_dict['mc_values'], np.sqrt(ratio_dict['data_values']), np.sqrt(ratio_dict['w2']))
    plot_ratio(
        ratio_dict['ratio'], axs[1], 
        numer_err=ratio_err,
        denom_err=None,
        hist_axis=data_hist.axes
    )
    
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
    mc_dir_lists = {
        'preEE': None,
        'postEE': None
        # Need to add other data eras eventually (2023, etc)
    }
    get_mc_dir_lists(mc_dir_lists)
    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    # MC_pqs = make_mc_dict(mc_dir_lists)
    MC_pqs = {}
    
    for data_era, dir_list in mc_dir_lists.items():
        for dir_name in dir_list:
            for sample_type in ['nominal']:  # Ignores the scale-ups and scale-downs. Not computed in merger.py.
                print('======================== \n', dir_name+" started")
                dirpath = LPC_FILEPREFIX_SIM+'/'+data_era+'/'+dir_name+'/'+sample_type+'/*merged.parquet'
                sample = ak.concatenate([
                    ak.from_parquet(file) for file in glob.glob(dirpath)
                ])

                # perform necessary cuts to apply pre-selections
                sideband_cuts(sample)

                # MC_pqs[dir_name] = concatenate_records(
                #     MC_pqs[dir_name], slimmed_parquet(MC_EXTRA_VARS, sample)
                # )
                MC_pqs[dir_name] = slimmed_parquet(MC_EXTRA_VARS, sample)
                
                del sample
                print('======================== \n', dir_name+" finished")

    data_dir_lists = {
        'data': None
    }
    get_data_dir_lists(data_dir_lists)
    Data_pqs = {}

    for data_era, dir_list in data_dir_lists.items():
        for dir_name in dir_list:
            dirpath = LPC_FILEPREFIX_DATA+'/'+dir_name+'/*merged.parquet'
            sample = ak.concatenate([
                ak.from_parquet(file) for file in glob.glob(dirpath)
            ])

            # perform necessary cuts to enter ttH enriched region
            sideband_cuts(sample)

            Data_pqs[dir_name] = slimmed_parquet(DATA_EXTRA_VARS, sample)
            
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