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
# lpc_filegroup = lambda s: f'Run3_{s}_merged'
lpc_filegroup = lambda s: f'Run3_{s}_merged_MultiBDT_output_mvaIDCorr_22_23'
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, lpc_filegroup('2022'), 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, lpc_filegroup('2023'), 'sim', '')
LPC_FILEPREFIX_24 = os.path.join(lpc_fileprefix, lpc_filegroup('2024'), 'sim', '')
END_FILEPATH = '*output.parquet' if re.search('MultiBDT_output', LPC_FILEPREFIX_22) is not None else '*merged.parquet'

DESTDIR = 'syst_unc_plots'
if not os.path.exists(DESTDIR):
    os.makedirs(DESTDIR)
FORCE_RERUN = True

APPLY_WEIGHTS = False
EVAL_CATEGORIES = True
EVAL_METHOD = '2D'
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
    "bbHToGG": r"$b\bar{b}H\rightarrow\gamma\gamma$",
    # signal
    "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$",
    "VBFToHH": r"VBF $HH\rightarrow bb\gamma\gamma$",
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
    # jet variables #
    'nonRes_lead_bjet_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'nonRes_sublead_bjet_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'sublead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    
    # dijet variables #
    'dijet_PNetRegMass': hist.axis.Regular(24, 70., 190., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False),
    
    # diphoton variables #
    'mass': hist.axis.Regular(40, 115., 135., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False),
    
    # BDT output #
    'MultiBDT_output': hist.axis.Regular(100, 0., 1., name='var', label=r'Multiclass BDT output', growth=False, underflow=False, overflow=False),
}
EXTRA_VARIABLES = {
    'eventWeight', MC_DATA_MASK, 'MultiBDT_output'
}

OPTIMIZED_CUTS = {
    '1D': [0.9977, 0.9946, 0.9874],
    '2D': [
        [0.987, 0.9982],
        [0.92, 0.994],
        [0.92, 0.9864],
    ]
}

XS = {
        # signal #
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_HH_c
        'GluGluToHH': 34.43*0.0026,
        
        # resonant background #
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#gluon_gluon_Fusion_Process
        'GluGluHToGG': 48520*0.00228,

        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ttH_Process
        'ttHToGG': 506.5*0.00228,
        
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#VBF_Process
        'VBFHToGG': 3779*0.00228,

        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#WH_Process + https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
        'VHToGG': (1369 + 882.4)*0.00228,
        
        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#bbH_Process
        'bbHToGG': 526.5*0.00228,

        # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Current_recommendations_for_di_H
        'VBFToHH': 1.870*0.0026
}

WEIGHT_SYSTS = [  # Up and Down
    'ElectronVetoSF', 'PreselSF', 'TriggerSF', 'Pileup'
    'bTagSF_sys_lf', 
    'bTagSF_sys_lfstats1', 'bTagSF_sys_lfstats2',
    'bTagSF_sys_cferr1', 'bTagSF_sys_cferr2', 
    'bTagSF_sys_hf', 
    'bTagSF_sys_hfstats1', 'bTagSF_sys_hfstats2',
    'bTagSF_sys_jes', 
]

VARIATION_SYSTS = [  # _up and _down
    'Et_dependent_ScaleEB', 'Et_dependent_ScaleEE', 
    'Et_dependent_Smearing', 
    'jec_syst_Total', 'jer_syst'
]

def sideband_cuts(sample):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = (
        sample['is_nonRes']
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
        dir_lists[sim_era] = list(os.listdir(sim_era))
        dir_lists[sim_era].sort()

def find_dirname(dir_name):
    sample_name_map = {
        # ggf HH (signal)
        'GluGluToHH': 'GluGluToHH',
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': 'GluGluToHH',
        'GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00': 'GluGluToHH',
        # # prompt-prompt non-resonant
        # 'GGJets': 'GGJets', 
        # # prompt-fake non-resonant
        # 'GJetPt20To40': 'GJetPt20To40', 
        # 'GJetPt40': 'GJetPt40', 
        # ggf H
        'GluGluHToGG': 'GluGluHToGG',
        'GluGluHToGG_M_125': 'GluGluHToGG',
        'GluGluHtoGG': 'GluGluHToGG',
        # ttH
        'ttHToGG': 'ttHToGG',
        'ttHtoGG_M_125': 'ttHToGG',
        'ttHtoGG': 'ttHToGG',
        # vbf H
        'VBFHToGG': 'VBFHToGG',
        'VBFHToGG_M_125': 'VBFHToGG',
        'VBFHtoGG': 'VBFHToGG',
        # VH
        'VHToGG': 'VHToGG',
        'VHtoGG_M_125': 'VHToGG',
        'VHtoGG': 'VHToGG',
        'VHtoGG_M-125': 'VHToGG',
        # bbH
        'BBHto2G_M_125': 'bbHToGG',
        'bbHtoGG': 'bbHToGG',
        # vbf HH
        'VBFHHto2B2G_CV_1_C2V_1_C3_1': 'VBFToHH'
    }
    if dir_name in sample_name_map:
        return sample_name_map[dir_name]
    else:
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

def generate_hists(pq_dict: dict, variable: str, axis, weight=APPLY_WEIGHTS):
    # https://indico.cern.ch/event/1433936/ #
    # Generate syst hists and ratio hists
    syst_hists = {}
    for ak_name, ak_arr in pq_dict.items():
        ak_hist = ak_arr[:, 0] if 'MultiBDT_output' in VARIABLES else ak_arr

        mask = ak_hist[MC_DATA_MASK]

        if weight:
            syst_hists[ak_name] = hist.Hist(axis, storage='weight').fill(
                var=ak_hist[variable][mask],
                weight=ak_hist['eventWeight'][mask],
            )
        else:
            syst_hists[ak_name] = hist.Hist(axis).fill(
                var=ak_hist[variable][mask]
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
        numer_values = ak_hist.values()
        denom_values = syst_hists["nominal"].values()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio_dict[f'{ratio_type}_ratio_values'] = numer_values / denom_values
            ratio_dict[f'{ratio_type}_ratio_err'] = ratio_error(
                numer_values, denom_values, 
                ak_hist.variances() if weight else np.sqrt(numer_values),
                syst_hists["nominal"].variances() if weight else np.sqrt(denom_values)
            )

    return syst_hists, ratio_dict

def plot_ratio(ratio, mpl_ax, hist_axis, numer_err=None, denom_err=None, central_value=1.0, color='black', lw=2.):
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

    mpl_ax.set_ylim(0.8, 1.2)
    # if np.min(ratio - numer_err) > 0.8 and np.max(ratio + numer_err) < 1.2:
    #     mpl_ax.set_ylim(0.8, 1.2)
    mpl_ax.axhline(
        central_value, color="black", linestyle="solid", lw=1.
    )
    mpl_ax.errorbar(
        hist_axis.centers[0], ratio, yerr=numer_err, 
        fmt='none', lw=lw, color=color, alpha=0.8
    )
    mpl_ax.stairs(
        ratio, edges=hist_axis.edges[0], fill=False, 
        baseline=1., lw=lw, color=color, alpha=0.8
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

def compute_uncertainty(syst_hists: dict, syst_name):
    nominal_integral = np.sum(syst_hists["nominal"].values())

    up_integral = np.sum(syst_hists[syst_name+"_up"].values())
    down_integral = np.sum(syst_hists[syst_name+"_down"].values())

    up_percent_diff = (up_integral - nominal_integral) / nominal_integral
    down_percent_diff = (down_integral - nominal_integral) / nominal_integral

    return {
        'up_percent_diff': float(up_percent_diff),
        'down_percent_diff': float(down_percent_diff),
        'avg_percent_diff': float( (np.abs(up_percent_diff) + np.abs(down_percent_diff)) / 2 )
    }

def get_ttH_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])

def get_QCD_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])

def cut_ak_dict(ak_dict: dict, index: int):
    return_ak_dict = copy.deepcopy(ak_dict)

    if EVAL_METHOD == '1D':

        lower_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index]
        if index > 0:
            upper_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index - 1]
        else:
            upper_cut_value = 1.0
        
        for ak_name in return_ak_dict.keys():
            mask = (
                return_ak_dict[ak_name]['MultiBDT_output'] > lower_cut_value
            ) & (
                return_ak_dict[ak_name]['MultiBDT_output'] <= upper_cut_value
            )
            return_ak_dict[ak_name] = return_ak_dict[ak_name][mask]

        return return_ak_dict
    
    elif EVAL_METHOD == '2D':

        lower_ttH_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index][0]
        lower_QCD_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index][1]
        if index > 0:
            upper_QCD_cut_value = OPTIMIZED_CUTS[EVAL_METHOD][index - 1][1]
        else:
            upper_QCD_cut_value = 1.
        
        for ak_name in return_ak_dict.keys():
            mask = (
                get_ttH_score(return_ak_dict[ak_name]['MultiBDT_output']) > lower_ttH_cut_value
            ) & (
                get_QCD_score(return_ak_dict[ak_name]['MultiBDT_output']) > lower_QCD_cut_value
            ) & (
                get_QCD_score(return_ak_dict[ak_name]['MultiBDT_output']) <= upper_QCD_cut_value
            )

            return_ak_dict[ak_name] = return_ak_dict[ak_name][mask]

        return return_ak_dict
    
    else:
        raise Exception(f"Method {EVAL_METHOD} not implemented yet.")

def plot(
    variable: str, syst_hists: dict, ratio_dict: dict, 
    year='2022', era='postEE', lumi=0.0, sample_name='signal',
    systname='smear', rel_dirpath='', weight=APPLY_WEIGHTS
):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8)
    )
    linewidth=2.
    for syst_name, syst_hist in syst_hists.items():
        hep.histplot(
            syst_hist, label=syst_name, 
            yerr=syst_hist.variances() if weight else True,
            ax=axs[0], lw=linewidth, histtype="step", alpha=0.8
        )
    
    for idx, ratio_type in enumerate(["up", "down"]):
        plot_ratio(
            ratio_dict[f'{ratio_type}_ratio_values'], axs[1], 
            numer_err=ratio_dict[f'{ratio_type}_ratio_err'],
            denom_err=None, hist_axis=syst_hists["nominal"].axes,
            color=cmap_petroff10[idx+1], lw=linewidth
        )
    
    # Plotting niceties #
    hep.cms.lumitext(f"{year}{era} {lumi:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend()
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(sample_name+'  '+syst_hists["nominal"].axes.label[0])
    axs[0].set_yscale('log') if variable == 'MultiBDT_output' else axs[0].set_yscale('linear')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{variable}_{systname}_up_down_variation{"_weighted" if weight else ""}.pdf')
    plt.savefig(f'{destdir}1dhist_{variable}_{systname}_up_down_variation{"_weighted" if weight else ""}.png')
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
    uncertainty_value = {}
    MC_pqs_merged = {}
    uncertainty_value_merged = {}
    
    for data_era, dir_list in mc_dir_lists.items():
        cut_era = data_era[data_era[:-1].rfind('/')+1:-1]
        year = data_era[data_era.find('Run3_202')+len('Run3_'):data_era.find('Run3_202')+len('Run3_202x')]
        print('======================== \n', year+''+cut_era+" started")
        MC_pqs[cut_era] = {}
        uncertainty_value[cut_era] = {}

        for dir_name in dir_list:
            std_dirname = find_dirname(dir_name)
            if std_dirname is None:
                print(f'{dir_name} not in samples selected for this computation.')
                continue
            if len(os.listdir(os.path.join(data_era, dir_name))) == 1:
                print(f'{dir_name} does not have variations computed.')
                continue
            if re.search('H', dir_name.upper()) is None:
                print(f'{dir_name} is non-resonant sample. No systematics will be computed as the non-resonant comes from data.')
                continue
            print('======================== \n', std_dirname+" started")
            MC_pqs[cut_era][std_dirname] = {}
            uncertainty_value[cut_era][std_dirname] = {}

            nominal_dirpath = os.path.join(data_era, dir_name, 'nominal', END_FILEPATH)
            nominal_sample = ak.from_parquet(glob.glob(nominal_dirpath)[0])
            sideband_cuts(nominal_sample)

            for weight_syst_name in WEIGHT_SYSTS:
                print('======================== \n', weight_syst_name+" started")

                crosssec_x_lumi =  XS[std_dirname] * LUMINOSITIES[data_era]

                syst_up_sample = copy.deepcopy(nominal_sample)
                syst_up_sample['eventWeight'] = syst_up_sample[f'weight_{weight_syst_name}Up'] * crosssec_x_lumi
                syst_down_sample = copy.deepcopy(nominal_sample)
                syst_down_sample['eventWeight'] = syst_up_sample[f'weight_{weight_syst_name}Down'] * crosssec_x_lumi

                MC_pqs[cut_era][std_dirname][weight_syst_name] = {
                    "nominal": slimmed_parquet(nominal_sample),
                    weight_syst_name+"_up": slimmed_parquet(syst_up_sample),
                    weight_syst_name+"_down": slimmed_parquet(syst_down_sample)
                }

                del syst_up_sample, syst_down_sample
                print('======================== \n', weight_syst_name+" finished")
                    
                # Ploting over variables for MC and Data
                for variable, axis in VARIABLES.items():
                    syst_hists, ratio_hists = generate_hists(
                        MC_pqs[cut_era][std_dirname][weight_syst_name], variable, axis,
                        weight=True
                    )
                    plot_dirpath = os.path.join(year, cut_era, std_dirname, '')
                    plot(
                        variable, syst_hists, ratio_hists, 
                        era=cut_era, year=year, lumi=LUMINOSITIES[data_era], 
                        sample_name=MC_NAMES_PRETTY[std_dirname], systname=weight_syst_name,
                        rel_dirpath=plot_dirpath, weight=True
                    )

                    if variable == 'mass':
                        uncertainty_value[cut_era][std_dirname][weight_syst_name] = compute_uncertainty(syst_hists, weight_syst_name)

            for syst_name in VARIATION_SYSTS:
                print('======================== \n', syst_name+" started")

                syst_up_dirpath = os.path.join(data_era, dir_name, syst_name+'_up', END_FILEPATH)
                syst_down_dirpath = os.path.join(data_era, dir_name, syst_name+'_down', END_FILEPATH)

                syst_up_sample = ak.from_parquet(glob.glob(syst_up_dirpath)[0])
                sideband_cuts(syst_up_sample)
                syst_down_sample = ak.from_parquet(glob.glob(syst_down_dirpath)[0])
                sideband_cuts(syst_down_sample)

                MC_pqs[cut_era][std_dirname][syst_name] = {
                    "nominal": slimmed_parquet(nominal_sample),
                    syst_name+"_up": slimmed_parquet(syst_up_sample),
                    syst_name+"_down": slimmed_parquet(syst_down_sample)
                }

                del syst_up_sample, syst_down_sample
                print('======================== \n', syst_name+" finished")

                # Ploting over variables for MC and Data
                for variable, axis in VARIABLES.items():
                    syst_hists, ratio_hists = generate_hists(
                        MC_pqs[cut_era][std_dirname][syst_name], variable, axis
                    )
                    plot_dirpath = os.path.join(year, cut_era, std_dirname, '')
                    plot(
                        variable, syst_hists, ratio_hists, 
                        era=cut_era, year=year, lumi=LUMINOSITIES[data_era], 
                        sample_name=MC_NAMES_PRETTY[std_dirname], systname=syst_name,
                        rel_dirpath=plot_dirpath
                    )

                    if variable == 'mass':
                        uncertainty_value[cut_era][std_dirname][syst_name] = compute_uncertainty(syst_hists, syst_name)

            print('======================== \n', std_dirname+" finished")
        
        for std_dirname, dir_systs in MC_pqs[cut_era].items():
            if std_dirname not in MC_pqs_merged:
                MC_pqs_merged[std_dirname] = {}
                uncertainty_value_merged[std_dirname] = {}

            for syst_name, syst_ak_dict in dir_systs.items():
                if syst_name not in MC_pqs_merged[std_dirname]:
                    MC_pqs_merged[std_dirname][syst_name] = copy.deepcopy(syst_ak_dict)
                else:
                    for syst_ak_name, syst_ak in syst_ak_dict.items():
                        MC_pqs_merged[std_dirname][syst_name][syst_ak_name] = concatenate_records(
                            MC_pqs_merged[std_dirname][syst_name][syst_ak_name], syst_ak
                        )

    for std_dirname, dir_systs in MC_pqs_merged.items():

        for syst_name, syst_ak_dict in dir_systs.items():

            for variable, axis in VARIABLES.items():
                syst_hists, ratio_hists = generate_hists(
                    syst_ak_dict, variable, axis,
                    weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                )
                plot_dirpath = os.path.join(std_dirname, '')
                plot(
                    variable, syst_hists, ratio_hists, 
                    era='', year='2022+2023', lumi=LUMINOSITIES['total_lumi'], 
                    sample_name=MC_NAMES_PRETTY[std_dirname], systname=syst_name,
                    rel_dirpath=plot_dirpath, 
                    weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                )

                if variable == 'mass':
                    uncertainty_value_merged[std_dirname][syst_name] = compute_uncertainty(syst_hists, syst_name)

    if not FORCE_RERUN and os.path.exists(os.path.join(DESTDIR, "uncertainties.json")):
        new_unc = copy.deepcopy(uncertainty_value)
        uncertainty_value = json.load(open(os.path.join(DESTDIR, "uncertainties.json")))

        for era in new_unc.keys():
            for dirname in new_unc[era].keys():
                for systname in new_unc[era][dirname].keys():
                    for varname in new_unc[era][dirname][systname].keys():

                        if era not in uncertainty_value:
                            uncertainty_value[era] = copy.deepcopy(new_unc[era])
                            continue
                        elif dirname not in uncertainty_value[era]:
                            uncertainty_value[era][dirname] = copy.deepcopy(new_unc[era][dirname])
                            continue
                        elif systname not in uncertainty_value[era][dirname]:
                            uncertainty_value[era][dirname][systname] = copy.deepcopy(new_unc[era][dirname][systname])
                            continue
                        elif varname not in uncertainty_value[era][dirname][systname]:
                            uncertainty_value[era][dirname][systname][varname] = copy.deepcopy(new_unc[era][dirname][systname][varname])
                            continue  
    with open(os.path.join(DESTDIR, "uncertainties.json"), "w") as f:
        json.dump(uncertainty_value, f)

    if not FORCE_RERUN and os.path.exists(os.path.join(DESTDIR, "uncertainties_merged.json")):
        new_unc = copy.deepcopy(uncertainty_value_merged)
        uncertainty_value_merged = json.load(open(os.path.join(DESTDIR, "uncertainties_merged.json")))

        for dirname in new_unc.keys():
            for systname in new_unc[dirname].keys():
                for varname in new_unc[dirname][systname].keys():

                    if dirname not in uncertainty_value_merged:
                        uncertainty_value_merged[dirname] = copy.deepcopy(new_unc[dirname])
                        continue
                    elif systname not in uncertainty_value_merged[dirname]:
                        uncertainty_value_merged[dirname][systname] = copy.deepcopy(new_unc[dirname][systname])
                        continue
                    elif varname not in uncertainty_value_merged[dirname][systname]:
                        uncertainty_value_merged[dirname][systname][varname] = copy.deepcopy(new_unc[dirname][systname][varname])
                        continue       
    with open(os.path.join(DESTDIR, "uncertainties_merged.json"), "w") as f:
        json.dump(uncertainty_value_merged, f)

    if EVAL_CATEGORIES:

        uncertainty_value_cat = {}
        uncertainty_value_cat_merged = {}

        for cat_idx, cat in enumerate(OPTIMIZED_CUTS[EVAL_METHOD]):

            # Unmerged era uncertainties
            uncertainty_value_cat[cat_idx] = {}

            for data_era in mc_dir_lists.keys():

                cut_era = data_era[data_era[:-1].rfind('/')+1:-1]
                year = data_era[data_era.find('Run3_202')+len('Run3_'):data_era.find('Run3_202')+len('Run3_202x')]
                uncertainty_value_cat[cat_idx][cut_era] = {}

                data_era_dict = MC_pqs[cut_era]

                for std_dirname, dir_systs in data_era_dict.items():

                    uncertainty_value_cat[cat_idx][cut_era][std_dirname] = {}

                    for syst_name, syst_ak_dict in dir_systs.items():
                        
                        cut_syst_ak_dict = cut_ak_dict(syst_ak_dict, cat_idx)

                        for variable, axis in VARIABLES.items():

                            syst_hists, ratio_hists = generate_hists(
                                cut_syst_ak_dict, variable, axis,
                                weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                            )
                            plot_dirpath = os.path.join(f'Cat{cat_idx}', year, cut_era, std_dirname, '')

                            plot(
                                variable, syst_hists, ratio_hists, 
                                era=cut_era, year=year, lumi=LUMINOSITIES[data_era], 
                                sample_name=MC_NAMES_PRETTY[std_dirname], systname=syst_name,
                                rel_dirpath=plot_dirpath,
                                weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                            )

                            if variable == 'mass':
                                uncertainty_value_cat[cat_idx][cut_era][std_dirname][syst_name] = compute_uncertainty(syst_hists, syst_name)

            # Merged era uncertainties
            uncertainty_value_cat_merged[cat_idx] = {}

            for std_dirname, dir_systs in MC_pqs_merged.items():

                uncertainty_value_cat_merged[cat_idx][std_dirname] = {}

                for syst_name, syst_ak_dict in dir_systs.items():

                    cut_syst_ak_dict = cut_ak_dict(syst_ak_dict, cat_idx)

                    for variable, axis in VARIABLES.items():
                        syst_hists, ratio_hists = generate_hists(
                            cut_syst_ak_dict, variable, axis,
                            weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                        )
                        plot_dirpath = os.path.join(f'Cat{cat_idx}', std_dirname, '')

                        plot(
                            variable, syst_hists, ratio_hists, 
                            era='', year='2022+2023', lumi=LUMINOSITIES['total_lumi'], 
                            sample_name=MC_NAMES_PRETTY[std_dirname], systname=syst_name,
                            rel_dirpath=plot_dirpath,
                            weight=True if syst_name in WEIGHT_SYSTS else APPLY_WEIGHTS
                        )

                        if variable == 'mass':
                            uncertainty_value_cat_merged[cat_idx][std_dirname][syst_name] = compute_uncertainty(syst_hists, syst_name)

        if not FORCE_RERUN and os.path.exists(os.path.join(DESTDIR, "uncertainties_cat.json")):
            new_unc = copy.deepcopy(uncertainty_value_cat)
            uncertainty_value_cat = json.load(open(os.path.join(DESTDIR, "uncertainties_cat.json")))

            for cat in new_unc.keys():
                for era in new_unc[cat].keys():
                    for dirname in new_unc[cat][era].keys():
                        for systname in new_unc[cat][era][dirname].keys():
                            for varname in new_unc[cat][era][dirname][systname].keys():

                                if cat not in uncertainty_value_cat:
                                    uncertainty_value_cat[cat] = copy.deepcopy(new_unc[cat])
                                    continue
                                elif era not in uncertainty_value_cat[cat]:
                                    uncertainty_value_cat[cat][era] = copy.deepcopy(new_unc[cat][era])
                                    continue
                                elif dirname not in uncertainty_value_cat[cat][era]:
                                    uncertainty_value_cat[cat][era][dirname] = copy.deepcopy(new_unc[cat][era][dirname])
                                    continue
                                elif systname not in uncertainty_value_cat[cat][era][dirname]:
                                    uncertainty_value_cat[cat][era][dirname][systname] = copy.deepcopy(new_unc[cat][era][dirname][systname])
                                    continue
                                elif varname not in uncertainty_value_cat[cat][era][dirname][systname]:
                                    uncertainty_value_cat[cat][era][dirname][systname][varname] = copy.deepcopy(new_unc[cat][era][dirname][systname][varname])
                                    continue  
        with open(os.path.join(DESTDIR, "uncertainties_cat.json"), "w") as f:
            json.dump(uncertainty_value_cat, f)

        if not FORCE_RERUN and os.path.exists(os.path.join(DESTDIR, "uncertainties_cat_merged.json")):
            new_unc = copy.deepcopy(uncertainty_value_cat_merged)
            uncertainty_value_cat_merged = json.load(open(os.path.join(DESTDIR, "uncertainties_cat_merged.json")))

            for cat in new_unc.keys():
                for dirname in new_unc[cat].keys():
                    for systname in new_unc[cat][dirname].keys():
                        for varname in new_unc[cat][dirname][systname].keys():

                            if cat not in uncertainty_value_cat_merged:
                                uncertainty_value_cat_merged[cat] = copy.deepcopy(new_unc[cat])
                                continue
                            elif dirname not in uncertainty_value_cat_merged[cat]:
                                uncertainty_value_cat_merged[cat][dirname] = copy.deepcopy(new_unc[cat][dirname])
                                continue
                            elif systname not in uncertainty_value_cat_merged[cat][dirname]:
                                uncertainty_value_cat_merged[cat][dirname][systname] = copy.deepcopy(new_unc[cat][dirname][systname])
                                continue
                            elif varname not in uncertainty_value_cat_merged[cat][dirname][systname]:
                                uncertainty_value_cat_merged[cat][dirname][systname][varname] = copy.deepcopy(new_unc[cat][dirname][systname][varname])
                                continue       
        with open(os.path.join(DESTDIR, "uncertainties_cat_merged.json"), "w") as f:
            json.dump(uncertainty_value_cat_merged, f)

if __name__ == '__main__':
    main()