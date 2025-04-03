import copy
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

lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/"
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, 'Run3_2022_merged', 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, 'Run3_2023_merged', 'sim', '')

DESTDIR = 'syst_unc_plots'
if not os.path.exists(DESTDIR):
    os.makedirs(DESTDIR)

APPLY_WEIGHTS = True
MC_DATA_MASK = 'MC_Data_mask'
FILL_VALUE = -999
MC_NAMES_PRETTY = {
    # non-resonant
    "GGJets": r"$\gamma\gamma+3j$",
    "GJetPt20To40": r"$\gamma+j$, 20<$p_T$<40GeV",
    "GJetPt40": r"$\gamma+j$, 40GeV<$p_T$",
    # single-H
    "GluGluHToGG": r"ggF $H\rightarrow \gamma\gamma$",
    "VBFHToGG": r"VBF $H\rightarrow \gamma\gamma$",
    "VHToGG": r"V$H\rightarrow\gamma\gamma$",
    "ttHToGG": r"$t\bar{t}H\rightarrow\gamma\gamma$",
    'bbHtoGG': r"$b\bar{b}H\rightarrow\gamma\gamma$",
    # signal
    "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$",
}
LUMINOSITIES = {
    os.path.join(LPC_FILEPREFIX_22, "preEE", ""): 7.9804,
    os.path.join(LPC_FILEPREFIX_22, "postEE", ""): 26.6717,
    os.path.join(LPC_FILEPREFIX_23, "preBPix", ""): 17.794,
    os.path.join(LPC_FILEPREFIX_23, "postBPix", ""): 9.451,
    # os.path.join(lpc_fileprefix, "Run3_2024", "sim", "2024", ""): 109.08,
}
LUMINOSITIES['total_lumi'] = sum(LUMINOSITIES.values())

# Dictionary of variables
VARIABLES = {
    # key: hist.axis axes for plotting #
    # # MET variables
    # 'puppiMET_sumEt': hist.axis.Regular(40, 150., 2000, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    # # jet-photon variables
    # 'nonRes_DeltaR_jg_min': hist.axis.Regular(30, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # # jet variables
    # 'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    # # ATLAS variables #
    # 'RegPt_balance': hist.axis.Regular(100, 0., 2., name='var', label=r'$HH p_{T} / (\gamma1 p_{T} + \gamma2 p_{T} + j1 p_{T} + j2 p_{T})$', growth=False, underflow=False, overflow=False), 
    # # photon variables
    # 'lead_mvaID_run3': hist.axis.Regular(100, -1., 1, name='var', label=r'lead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False), 
    # 'sublead_mvaID_run3': hist.axis.Regular(100, -1., 1, name='var', label=r'sublead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False), 
    # # dijet variables
    # 'dijet_PNetRegMass': hist.axis.Regular(24, 70., 190., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False)
    'mass': hist.axis.Regular(20, 75., 175., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False)
}
EXTRA_VARIABLES = {
    'eventWeight'
}

def sideband_cuts(sample):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = (
        sample['nonRes_has_two_btagged_jets'] 
        & sample['is_nonRes']
        & sample['fiducialGeometricFlag']
    )
    sample[MC_DATA_MASK] = event_mask

def get_mc_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    
    # Pull MC sample dir_list
    for sim_era in dir_lists.keys():
        dir_list = list(os.listdir(sim_era))

        for dir_name in dir_list:
            if (
                re.search('up', dir_name.lower()) is None
                and re.search('down', dir_name.lower()) is None
                and re.search('nominal', dir_name.lower()) is None
            ):
                dir_list.remove(dir_name)

        dir_list.sort()
        dir_lists[sim_era] = copy.deepcopy(dir_list)

def find_dirname(dir_name):
    for std_sample_name in MC_NAMES_PRETTY.keys():
        if std_sample_name[:3].lower() == dir_name[:3].lower():
            return std_sample_name
    return None

def slimmed_parquet(sample):
    """
    Creates a new slim parquet.
    """
    return ak.zip(
        {field: sample[field] for field in (set(VARIABLES.keys()) | EXTRA_VARIABLES)}
    )

def concatenate_records(base_sample, added_sample):
    """
    Extrapolates the ak.concatenate() functionality to copy across fields.
    """
    return ak.zip(
        {
            field: ak.concatenate((base_sample[field], added_sample[field])) for field in base_sample.fields
        }
    )

def generate_hists(pq_dict: dict, variable: str, axis):
    # https://indico.cern.ch/event/1433936/ #
    # Generate syst hists and ratio hists
    syst_hists = {}
    for ak_name, ak_arr in pq_dict.items():

        if APPLY_WEIGHTS:
            syst_hists[ak_name] = hist.Hist(axis, storage='weight').fill(
                var=ak_arr[variable],
                weight=ak_arr['eventWeight']
            )
        else:
            syst_hists[ak_name] = hist.Hist(axis).fill(
                var=ak_arr[variable]
            )

    # Generate ratio dict
    ratio_dict = {}
    for ak_name, ak_hist in syst_hists.items():
        if ak_name == "nominal":
            continue
        elif re.search("up", ak_name.lower()) is not None:
            ratio_type = 'up'
        elif re.search("down", ak_name.lower()) is not None:
            ratio_type = 'down'
        numer_values = ak_hist.values().flatten()
        denom_values = syst_hists["nominal"].values().flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio_dict[f'{ratio_type}_ratio_values'] = numer_values / denom_values
            ratio_dict[f'{ratio_type}_ratio_err'] = ratio_error(
                numer_values, denom_values, 
                ak_hist.variances().flatten(),
                syst_hists["nominal"].variances().flatten()
            )

    return syst_hists, ratio_dict

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

def plot(
    variable: str, syst_hists: dict, ratio_dict: dict, 
    year='2022', era='postEE', lumi=0.0, sample_name='signal',
    systname='smear', rel_dirpath=''
):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8)
    )
    for syst_name, syst_hist in syst_hists.items():
        hep.histplot(
            syst_hist, label=syst_name, w2=syst_hist.variances().flatten(), 
            ax=axs[0], linewidth=3, histtype="step"
        )
    
    for ratio_type in ["up", "down"]:
        plot_ratio(
            ratio_dict[f'{ratio_type}_ratio_values'], axs[1], 
            numer_err=ratio_dict[f'{ratio_type}_ratio_err'],
            denom_err=None, hist_axis=syst_hists["nominal"].axes
        )
    
    # Plotting niceties #
    hep.cms.lumitext(f"{year}{era} {lumi} (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend()
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(sample_name+'  '+syst_hists["nominal"].axes.label[0])
    # axs[0].set_yscale('log')
    axs[0].set_yscale('linear')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{systname}_up_down_variation.pdf')
    plt.savefig(f'{destdir}1dhist_{systname}_up_down_variation.png')
    plt.close()
    
def main():
    """
    Performs the Up/Down variation comparison.
    """
    mc_dir_lists = {
        os.path.join(LPC_FILEPREFIX_22, "preEE", ""): None,
        os.path.join(LPC_FILEPREFIX_22, "postEE", ""): None,
        os.path.join(LPC_FILEPREFIX_23, "preBPix", ""): None,
        os.path.join(LPC_FILEPREFIX_23, "postBPix", ""): None,
    }
    

    get_mc_dir_lists(mc_dir_lists)
    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    # MC_pqs = make_mc_dict(mc_dir_lists)
    MC_pqs = {}
    
    for data_era, dir_list in mc_dir_lists.items():
        cut_era = data_era[data_era[:-1].rfind('/')+1:-1]
        year = data_era[data_era.find('Run3_202')+len('Run3_'):data_era.find('Run3_202')+len('Run3_202x')]
        print('======================== \n', year+''+cut_era+" started")
        MC_pqs[data_era] = {}

        for dir_name in dir_list:
            std_dirname = find_dirname(dir_name)
            if std_dirname is None:
                continue
            print('======================== \n', std_dirname+" started")
            MC_pqs[data_era][std_dirname] = {}

            nominal_dirpath = os.path.join(data_era, dir_name, 'nominal', '*merged.parquet')
            nominal_sample = ak.from_parquet(glob.glob(nominal_dirpath)[0])
            sideband_cuts(nominal_sample)

            for syst_name in ['Et_dependent_ScaleEB', 'Et_dependent_ScaleEE', 'Et_dependent_Smearing', 'jec_syst_Total', 'jer_syst']:
                print('======================== \n', syst_name+" started")

                syst_up_dirpath = os.path.join(data_era, dir_name, syst_name+'_up', '*merged.parquet')
                syst_down_dirpath = os.path.join(data_era, dir_name, syst_name+'_down', '*merged.parquet')

                syst_up_sample = ak.from_parquet(glob.glob(syst_up_dirpath)[0])
                sideband_cuts(syst_up_sample)
                syst_down_sample = ak.from_parquet(glob.glob(syst_down_dirpath)[0])
                sideband_cuts(syst_down_sample)

                MC_pqs[data_era][std_dirname][syst_name] = {
                    "nominal": slimmed_parquet(nominal_sample),
                    syst_name+"up": slimmed_parquet(syst_up_sample),
                    syst_name+"down": slimmed_parquet(syst_down_sample)
                }

                del syst_up_sample, syst_down_sample
                print('======================== \n', syst_name+" finished")

                # Ploting over variables for MC and Data
                for variable, axis in VARIABLES.items():
                    syst_hists, ratio_hists = generate_hists(MC_pqs[data_era][std_dirname][syst_name], variable, axis)
                    plot_dirpath = os.path.join(year, cut_era, std_dirname, '')
                    plot(
                        variable, syst_hists, ratio_hists, 
                        era=cut_era, year=year, lumi=LUMINOSITIES[data_era], 
                        sample_name=MC_NAMES_PRETTY[std_dirname], systname=syst_name,
                        rel_dirpath=plot_dirpath
                    )

            print('======================== \n', std_dirname+" finished")

            

if __name__ == '__main__':
    main()