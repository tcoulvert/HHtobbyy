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
lpc_filegroup = lambda s: f'Run3_{s}_mergedBoosted'
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, lpc_filegroup('2022'), 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, lpc_filegroup('2023'), 'sim', '')
LPC_FILEPREFIX_24 = os.path.join(lpc_fileprefix, lpc_filegroup('2024'), 'sim', '')
END_FILEPATH = '*output.parquet' if re.search('MultiBDT_output', LPC_FILEPREFIX_22) is not None else '*merged.parquet'

DESTDIR = 'boosted_HHbbgg_plots'
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
    "bbHToGG": r"$b\bar{b}H\rightarrow\gamma\gamma$",
    # signal
    "GluGluToHH": r"ggF $HH\rightarrow bb\gamma\gamma$",
}
LUMINOSITIES = {
    "2022preEE": 7.9804,
    "2022postEE": 26.6717,
    "2023preBPix": 17.794,
    "2023postBPix": 9.451,
    "2024": 109.08,
}
LUMINOSITIES['total_lumi'] = sum(LUMINOSITIES.values())

# Dictionary of variables
VARIABLES = {
    # key: hist.axis axes for plotting #
    # fatjet variables
    # 'fatjet1_pt': hist.axis.Regular(40, 20., 2000, name='var', label=r'lead fatjet $p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'fatjet1_mass': hist.axis.Regular(75, 50., 200, name='var', label=r'lead fatjet mass [GeV]', growth=False, underflow=False, overflow=False),

    # diphoton variables
    # 'pt': hist.axis.Regular(40, 20., 2000, name='var', label=r'$\gamma\gamma$ $p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),

    # single photon variables
    # 'lead_pt': hist.axis.Regular(40, 20., 500., name='var', label=r'lead $\gamma$ $p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'sublead_pt': hist.axis.Regular(40, 20., 500., name='var', label=r'sublead $\gamma$ $p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'lead_mvaID': hist.axis.Regular(20, -1., 1., name='var', label=r'lead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False),
    # 'sublead_mvaID': hist.axis.Regular(20, -1., 1., name='var', label=r'sublead $\gamma$ MVA ID', growth=False, underflow=False, overflow=False),
    # 'Res_CosThetaStar_gg': hist.axis.Regular(20, -1., 1., name='var', label=r'Cos($\theta_{gg}^*$)', growth=False, underflow=False, overflow=False),
}
BLINDED_VARIABLES = {
    # diphoton variables
    'mass': (
        hist.axis.Regular(55, 80., 180., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False),
        [115, 135]
    )
}
EXTRA_MC_VARIABLES = {
    'eventWeight', 
    'dZ', 'mass', 'HH_PNetRegMass',
    'weight_ElectronVetoSF', 'weight_PreselSF', 'weight_TriggerSF', 'weight_Pileup',
    'weight_bTagSF_sys_lf', 
    'weight_bTagSF_sys_lfstats1', 'weight_bTagSF_sys_lfstats2',
    'weight_bTagSF_sys_hf', 
    'weight_bTagSF_sys_hfstats1', 'weight_bTagSF_sys_hfstats2',
    MC_DATA_MASK
}
EXTRA_DATA_VARIABLES = {
    'dZ', 'mass', 'HH_PNetRegMass',
    MC_DATA_MASK
}

VARIATION_SYSTS = [  # _up and _down
    'Et_dependent_ScaleEB', 'Et_dependent_ScaleEE', 
    'Et_dependent_Smearing', 
    'jec_syst_Total', 'jer_syst'
]

def get_era(filepath):
    era = ''
    for sub_era in ['preEE', 'postEE', 'preBPix', 'postBPix']:
        if re.search(sub_era, filepath) is not None:
            era = sub_era
            break
    year = filepath[filepath.find('Run3_202')+len('Run3_'):filepath.find('Run3_202')+len('Run3_202x')]

    return year+era

def get_lumi(era):
    if era in LUMINOSITIES:
        return LUMINOSITIES[era]
    elif re.search('-', era) is None:
        lumi_total = 0
        for sub_era, sub_lumi in LUMINOSITIES.items():
            if re.search(sub_era, era) is not None: lumi_total += sub_lumi
    else:
        lumi_total = 0
        split_eras = era.split('-')
        for split_era in split_eras:
            for sub_era, sub_lumi in LUMINOSITIES.items():
                if re.search(sub_era, split_era) is not None: lumi_total += sub_lumi

def sideband_cuts(sample, pathway=0):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = (
        sample["Res_has_atleast_one_fatjet"]
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        ) & (  # fatjet cuts
            (sample['fatjet1_pt'] > 250)
            & (
                (sample['fatjet1_mass'] > 100)  # fatjet1_msoftdrop
                & (sample['fatjet1_mass'] < 160)
            ) & (sample['fatjet1_particleNet_XbbVsQCD'] > 0.8)
        ) & (  # good photon cuts (for boosted regime)
            (sample['lead_mvaID'] > 0.)
            & (sample['sublead_mvaID'] > 0.)
        )
        # ) & (sample['fatjet1_pt'] > 250)
        # )
    )

    # if pathway == 0:
    #     event_mask = event_mask & (
    #         (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90'])
    #         | (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95'])
    #     )
    # elif pathway == 1:
    #     event_mask = event_mask & (
    #         (sample['DiphotonMVA14p25_Mass90'])
    #         | (sample['DiphotonMVA14p25_Tight_Mass90'])
    #     )
    # elif pathway == 2:
    #     event_mask = event_mask & (
    #         (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90'])
    #         | (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95'])
    #     ) & ~(
    #         (sample['DiphotonMVA14p25_Mass90'])
    #         | (sample['DiphotonMVA14p25_Tight_Mass90'])
    #     )
    # elif pathway == 3:
    #     event_mask = event_mask & (
    #         (sample['DiphotonMVA14p25_Mass90'])
    #         | (sample['DiphotonMVA14p25_Tight_Mass90'])
    #     ) & ~(
    #         (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90'])
    #         | (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95'])
    #     )

    sample[MC_DATA_MASK] = event_mask

    if 'eventWeight' in sample.fields:
        # print(f" {sample['sample_name'][0]} | pre-rescale yield: {np.sum(sample['eventWeight'][event_mask])}")
        if sample['sample_name'][0] == 'VBFToHH':
            sample['eventWeight'] = sample['eventWeight'] * (LUMINOSITIES['total_lumi'] / (LUMINOSITIES['total_lumi'] - LUMINOSITIES['2024'] - LUMINOSITIES['2022postEE']))
        sample['eventWeight'] = sample['eventWeight'] * (LUMINOSITIES['total_lumi'] / (LUMINOSITIES['total_lumi'] - LUMINOSITIES['2024']))
        # print(f" {sample['sample_name'][0]} | post-rescale yield: {np.sum(sample['eventWeight'][event_mask])}")

    if 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90' in sample.fields:
        sample[MC_DATA_MASK] = (sample[MC_DATA_MASK]) & (
            (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90'])  # old triggers
            & (sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95'])
        )

def get_mc_dir_lists(dir_lists: dict, all: bool=False):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    
    # Pull MC sample dir_list
    for sim_era in dir_lists.keys():
        if all:
            dir_lists[sim_era] = glob.glob(os.path.join(sim_era, "**", END_FILEPATH), recursive=True)
        else:
            var = "nominal"
            for sim_type in VARIATION_SYSTS: 
                if re.search(sim_type, sim_era) is not None:
                    var_direction = '_up' if re.search('_up', sim_era) else '_down'
                    var = sim_type + var_direction
                    break
            
            dir_lists[sim_era] = glob.glob(os.path.join(sim_era, "**", var, END_FILEPATH), recursive=True)
        dir_lists[sim_era].sort()

def get_data_dir_lists(dir_lists: dict):
    
    # Pull Data sample dir_list
    for data_era in dir_lists.keys():
        dir_lists[data_era] = glob.glob(os.path.join(data_era, "**", END_FILEPATH), recursive=True)
        dir_lists[data_era].sort()

def slimmed_parquet(sample, extra_variables):
    """
    Creates a new slim parquet.
    """
    return ak.zip(
        {field: sample[field] for field in (set(VARIABLES.keys()) | set(BLINDED_VARIABLES.keys()) | extra_variables) & set(sample.fields)}
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

def generate_hists(
    pq_dict: dict, variable: str, axis, blind_edges=None, density=False, ratio=False
):
    # https://indico.cern.ch/event/1433936/ #
    # Generate syst hists and ratio hists
    hists = {}
    for ak_name, ak_arr in pq_dict.items():
        if blind_edges is not None:
            mask = (
                (
                    (ak_arr[variable] < blind_edges[0]) 
                    | (ak_arr[variable] > blind_edges[1])
                ) & (ak_arr[MC_DATA_MASK])
            )
        else:
            mask = ak_arr[MC_DATA_MASK]

        if re.search('mc', ak_name.lower()) and APPLY_WEIGHTS:
            hists[ak_name] = hist.Hist(axis, storage='weight').fill(
                var=ak_arr[variable][mask],
                weight=ak_arr['eventWeight'][mask],
            )
        else:
            hists[ak_name] = hist.Hist(axis).fill(
                var=ak_arr[variable][mask]
            )

    return hists

def datamc_generate_hists(
    pq_dict: dict, variable: str, axis, blind_edges=None, density=False
):
    # https://indico.cern.ch/event/1433936/ #
    # Generate syst hists and ratio hists
    data_hist, mc_hists = None, {}

    for ak_name, ak_arr in pq_dict.items():
        if blind_edges is not None:
            mask = (
                (
                    (ak_arr[variable] < blind_edges[0]) 
                    | (ak_arr[variable] > blind_edges[1])
                ) & (ak_arr[MC_DATA_MASK])
            )
        else:
            mask = ak_arr[MC_DATA_MASK]

        if re.search('mc', ak_name.lower()) is not None and APPLY_WEIGHTS:
            mc_hists[ak_name] = hist.Hist(axis, storage='weight').fill(
                var=ak_arr[variable][mask],
                weight=ak_arr['eventWeight'][mask],
            )
        elif re.search('mc', ak_name.lower()) is not None:
            mc_hists[ak_name] = hist.Hist(axis).fill(
                var=ak_arr[variable][mask]
            )
        else:
            data_hist = hist.Hist(axis).fill(
                var=ak_arr[variable][mask]
            )
    
    # Generate ratio dict
    ratio_dict = {
        'numer_values': np.sum([mc_hist.values() for mc_hist in mc_hists.values()]),
        'numer_err': np.sum([mc_hist.variances() for mc_hist in mc_hists.values()]),
        'denom_values': data_hist.values(),
        'denom_err': np.sqrt(data_hist.values()),
    }
    if density:
        ratio_dict['numer_err'] = ratio_dict['numer_err'] / np.sum(ratio_dict['numer_values'])
        ratio_dict['denom_err'] = ratio_dict['denom_err'] / np.sum(ratio_dict['denom_values'])

        ratio_dict['numer_values'] = ratio_dict['numer_values'] / np.sum(ratio_dict['numer_values'])
        ratio_dict['denom_values'] = ratio_dict['denom_values'] / np.sum(ratio_dict['denom_values'])
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio_dict['ratio_values'] = ratio_dict['numer_values'] / ratio_dict['denom_values']
        ratio_dict['ratio_err'] = ratio_error(
            ratio_dict['numer_values'], ratio_dict['denom_values'], 
            ratio_dict['numer_err'], ratio_dict['denom_err']
        )

    return mc_hists, data_hist, ratio_dict

def plot_ratio(
    ratio, mpl_ax, hist_axis, 
    numer_err=None, denom_err=None, central_value=1.0, 
    color='black', lw=2.
):
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

    mpl_ax.set_ylim(0.25, 4.)
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

def datamc_plot(
    variable: str, mc_hists: dict, data_hist, ratio_dict: dict,
    era='2022postEE', lumi=0.0,
    rel_dirpath='', density=False,
):
    
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(20, 16)
    )
    linewidth=2.

    hep.histplot(
        list(mc_hists.values()), label=list(mc_hists.keys()), 
        yerr=np.vstack((np.tile(np.zeros_like(ratio_dict['numer_err']), (len(mc_hists)-1, 1)), ratio_dict['numer_err'])),
        # yerr=ratio_dict['numer_err']
        stack=True, ax=axs[0], linewidth=3, histtype="fill", sort="yield", density=density
    )
    
    hep.histplot(
        data_hist, label='Data', 
        yerr=True,
        ax=axs[0], lw=linewidth, histtype="errorbar", color="black",
        density=density
    )

    plot_ratio(
        ratio_dict['ratio_values'], axs[1], 
        numer_err=ratio_dict['ratio_err'],
        denom_err=None, hist_axis=data_hist.axes,
        lw=linewidth
    )
    
    # Plotting niceties #
    hep.cms.lumitext(f"{era} {lumi:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), reverse=True)
    plt.tight_layout(rect=(0, 0.01, 1, 1))
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(data_hist.axes.label[0])
    axs[0].set_yscale('log')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{variable}_DataMC.pdf')
    plt.savefig(f'{destdir}1dhist_{variable}_DataMC.png')
    plt.close()

def plot(
    variable: str, hists: dict, 
    era='2022postEE', lumi=0.0,
    rel_dirpath='', histtypes=None, density=False,
):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, ax = plt.subplots(figsize=(20, 16))
    linewidth=2.

    hist_names = list(hists.keys())
    if histtypes is None:
        histtypes = {hist_name: "step" for hist_name in hist_names}
    for hist_name in hist_names:
        hep.histplot(
            hists[hist_name], label=hist_name, 
            yerr=hists[hist_name].variances() if (APPLY_WEIGHTS and re.search('mc', hist_name.lower()) is not None) else True,
            ax=ax, lw=linewidth, histtype=histtypes[hist_name],
            density=density
        )

    # Plotting niceties #
    hep.cms.lumitext(f"{era} {lumi:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)
    # Plot legend properly
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), reverse=True)
    plt.tight_layout(rect=(0, 0.01, 1, 1))
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    ax.set_xlabel(hists[hist_names[0]].axes.label[0])
    ax.set_yscale('log')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{variable}_comparison.pdf')
    plt.savefig(f'{destdir}1dhist_{variable}_comparison.png')
    plt.close()

def get_concat_samples(sample_dirs: dict, save=False):
    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    sample_pqs = {}
    for dir_name, dir_dict in sample_dirs.items():
        print('======================== \n', dir_name+" started")
        sample_pqs[dir_name] = {}

        pathway = 0
        if re.search('MVATRG', dir_name) is not None:
            pathway = 1
        elif re.search('R9diffTRG', dir_name) is not None:
            pathway = 2
        elif re.search('MVAdiffTRG', dir_name) is not None:
            pathway = 3

        for sample_era, sample_list in dir_dict.items():
            print('======================== \n', sample_era+" started")
            sample_pqs[dir_name][sample_era] = []

            for samplefilepath in sample_list:
                print('======================== \n', samplefilepath[:samplefilepath.rfind('.')]+" started")

                sample = ak.from_parquet(samplefilepath)

                # if 'DiphotonMVA14p25_Mass90' not in sample.fields: 
                #     print('no MVA trg')
                #     continue

                sideband_cuts(sample, pathway=pathway)
                sample_pqs[dir_name][sample_era].append(
                    slimmed_parquet(
                        sample, 
                        EXTRA_MC_VARIABLES if re.search('mc', dir_name.lower()) is not None else EXTRA_DATA_VARIABLES
                    )
                )

                if save:
                    output_pq_filepath = (
                        samplefilepath[:samplefilepath.find('Run3_202')+len(lpc_filegroup('202x'))]
                        + '_Cut_output'
                        + samplefilepath[samplefilepath.find('Run3_202')+len(lpc_filegroup('202x')):]
                    )
                    if not os.path.exists(output_pq_filepath[:output_pq_filepath.rfind('/')]):
                        os.makedirs(output_pq_filepath[:output_pq_filepath.rfind('/')])
                    ak.to_parquet(sample[sample[MC_DATA_MASK]], output_pq_filepath)
                    print(f"======================== \nSaved out new file at:\n{output_pq_filepath}")

                del sample
                print('======================== \n', samplefilepath[:samplefilepath.rfind('.')]+" finished")

            print('======================== \n', sample_era+" finished")

        print('======================== \n', dir_name+" finished")

    # Concatenate the samples for comparison
    concat_samples = {}
    for dir_name, dir_dict in sample_pqs.items():
        concat_samples[dir_name] = None

        for sample_era, sample_list in dir_dict.items():

            for sample in sample_list:

                if concat_samples[dir_name] is None:
                    concat_samples[dir_name] = copy.deepcopy(sample)
                else:
                    concat_samples[dir_name] = concatenate_records(concat_samples[dir_name], sample)

    return concat_samples
    
def main(
    sample_dirs, save=False, plottype='comparison', density=False,
    era=None, lumi=None, all=False
):
    """
    Performs the sample comparison.
    """

    for dir_name, dir_dict in sample_dirs.items():
        if re.search('mc', dir_name.lower()) is not None:
            get_mc_dir_lists(dir_dict, all=all)
        elif re.search('data', dir_name.lower()) is not None:
            get_data_dir_lists(dir_dict)
        else:
            raise Exception(f"Ambiguous whether dirlist {dir_name} is Data or MC, please include either word in the name (key) of the dirlist.")
        
    concat_samples = get_concat_samples(sample_dirs, save=save)

    plot_list = []
    if plottype == 'split':
        for dir_name, dir_dict in concat_samples:
            plot_list.append({dir_name: dir_dict})
    else:
        plot_list.append(concat_samples)

    for concat_dict in plot_list:
        # Ploting over variables for MC and Data
        for variable, axis in VARIABLES.items():
            if plottype == 'Data/MC':
                mc_hists, data_hist, ratio_dict = datamc_generate_hists(
                    concat_dict, variable, axis, density=density,
                )
                datamc_plot(
                    variable, mc_hists, data_hist, ratio_dict, 
                    era=era, lumi=lumi, density=density
                )
            else:
                hists = generate_hists(
                    concat_dict, variable, axis, density=density,
                )
                plot(
                    variable, hists,
                    era=era, lumi=lumi, density=density
                )
        # Ploting over variables for MC and Data
        for variable, (axis, blind_edges) in BLINDED_VARIABLES.items():
            if plottype == 'Data/MC':
                mc_hists, data_hist, ratio_dict = datamc_generate_hists(
                    concat_dict, variable, axis,
                    blind_edges=blind_edges, density=density,
                )
                datamc_plot(
                    variable, mc_hists, data_hist, ratio_dict, 
                    era=era, lumi=lumi, density=density,
                )
            else:
                hists = generate_hists(
                    concat_dict, variable, axis,
                    blind_edges=blind_edges, density=density,
                )
                plot(
                    variable, hists, 
                    era=era, lumi=lumi, density=density
                )
            

if __name__ == '__main__':
    sample_dirs = {
        'Data-2022-24': {
            os.path.join(LPC_FILEPREFIX_22[:-len('sim/')], "data", ""): None,
            os.path.join(LPC_FILEPREFIX_23[:-len('sim/')], "data", ""): None,
            os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
        },
        # signal
        'MC-2022-24-GluGluToHH': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GluGluToHH", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
        },
        'MC-2022-24-VBFToHH': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
            # os.path.join(LPC_FILEPREFIX_22, "postEE", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        },
        # single H
        'MC-2022-24-GluGluHToGG': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GluGluHToGG_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GluGluHToGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "GluGluHtoGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "GluGluHtoGG", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GluGluHtoGG", ""): None,
        },
        'MC-2022-24-VBFHToGG': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "VBFHToGG_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "VBFHToGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "VBFHtoGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "VBFHtoGG", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "VBFHtoGG", ""): None,
        },
        'MC-2022-24-VHToGG': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "VHtoGG_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "VHtoGG_M-125", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "VHtoGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "VHtoGG", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "VHtoGG", ""): None,
        },
        'MC-2022-24-ttHToGG': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "ttHtoGG_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "ttHToGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "ttHtoGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "ttHtoGG", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "ttHtoGG", ""): None,
        },
        'MC-2022-24-bbHToGG': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "BBHto2G_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "BBHto2G_M_125", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "bbHtoGG", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "bbHtoGG", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "bbHtoGG", ""): None,
        },
        # non-resonant
        # 'MC-2022-24-GGJets': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "GGJets", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "GGJets", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "GGJets", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "GGJets", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "GGJets", ""): None,
        # },
        # 'MC-2022-24-GJetPt20To40': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "GJetPt20To40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "GJetPt20To40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "GJetPt20To40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "GJetPt20To40", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "GJetPt20To40", ""): None,
        # },
        # 'MC-2022-24-GJetPt40': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "GJetPt40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "GJetPt40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "GJetPt40", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "GJetPt40", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "GJetPt40", ""): None,
        # },
        # 'MC-2022-24-TTGG': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "TTGG", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "TTGG", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "TTGG", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "TTGG", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "GGJets", ""): None,
        # },
        # 'MC-2022-24-TTGJetPt10To100': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "TTG_1Jets_PTG_10to100", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "TTGJetPt10To100", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "TTGJetPt10To100", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "TTGJetPt10To100", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "TTGJetPt10To100", ""): None,
        # },
        # 'MC-2022-24-TTGJetPt100To200': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "TTG_1Jets_PTG_100to200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "TTGJetPt100To200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "TTGJetPt100To200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "TTGJetPt100To200", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "TTGJetPt100To200", ""): None,
        # },
        # 'MC-2022-24-TTGJetPt200': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "TTG_1Jets_PTG_200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_22, "postEE", "TTGJetPt200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "TTGJetPt200", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "TTGJetPt200", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "TTGJetPt200", ""): None,
        # },
    }

    main(
        sample_dirs, density=False,
        era="2022-24", lumi=LUMINOSITIES["total_lumi"],
        plottype='Data/MC', 
        all=True,
        save=True
    )

    # sample_dirs = {
    #     'Data-2024-R9TRG': {
    #         os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
    #     },
    #     'Data-2024-MVATRG': {
    #         os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
    #     },
    #     'Data-2024-R9diffTRG': {
    #         os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
    #     },
    #     'Data-2024-MVAdiffTRG': {
    #         os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
    #     },
    # }

    # main(
    #     sample_dirs, density=False,
    #     era="2024", lumi=LUMINOSITIES["2024"],
    #     # plottype='Data/MC',
    # )