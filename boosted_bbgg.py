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
    # # MET variables
    # 'puppiMET_sumEt': hist.axis.Regular(40, 150., 2000, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    # 'puppiMET_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'puppiMET $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 

    # # jet-MET variables
    # 'nonRes_DeltaPhi_j1MET': hist.axis.Regular(20,-3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    # 'nonRes_DeltaPhi_j2MET': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_2,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    # # jet-photon variables
    # 'nonRes_DeltaR_jg_min': hist.axis.Regular(30, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # # jet-lepton variables
    # 'DeltaR_b1l1': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=False, underflow=False, overflow=False),

    # # bjet variables
    # 'lead_bjet_PNetRegPt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'nonRes_lead_bjet_eta': hist.axis.Regular(20, -5., 5., name='var', label=r'lead bjet $\eta$', growth=False, underflow=False, overflow=False),
    # 'nonRes_lead_bjet_btagPNetB': hist.axis.Regular(50, 0., 1., name='var', label=r'$j_{lead}$ PNet btag score', growth=False, underflow=False, overflow=False), 
    # 'lead_bjet_RegPt_over_Mjj': hist.axis.Regular(50, 0., 4., name='var', label=r'$j1 p_{T} / M_{jj}$', growth=False, underflow=False, overflow=False), 
    # 'lead_bjet_sigmapT_over_RegPt': hist.axis.Regular(50, 0., 0.02, name='var', label=r'$j1 \sigma p_{T} / p_{T}$', growth=False, underflow=False, overflow=False), 
    # # ----------
    # 'sublead_bjet_PNetRegPt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'nonRes_sublead_bjet_eta': hist.axis.Regular(20, -5., 5., name='var', label=r'lead bjet $\eta$', growth=False, underflow=False, overflow=False),
    # 'nonRes_sublead_bjet_btagPNetB': hist.axis.Regular(50, 0., 1., name='var', label=r'$j_{lead}$ PNet btag score', growth=False, underflow=False, overflow=False), 
    # 'sublead_bjet_RegPt_over_Mjj': hist.axis.Regular(50, 0., 2., name='var', label=r'$j2 p_{T} / M_{jj}$', growth=False, underflow=False, overflow=False),
    # 'sublead_bjet_sigmapT_over_RegPt': hist.axis.Regular(50, 0., 0.02, name='var', label=r'$j2 \sigma p_{T} / p_{T}$', growth=False, underflow=False, overflow=False),

    # # jet variables
    # 'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    # 'nonRes_chi_t0': hist.axis.Regular(40, 0., 150, name='var', label=r'$\chi_{t0}^2$', growth=False, underflow=False, overflow=False), 
    # 'nonRes_chi_t1': hist.axis.Regular(30, 0., 500, name='var', label=r'$\chi_{t1}^2$', growth=False, underflow=False, overflow=False), 

    # # lepton variables
    # 'lepton1_pt': hist.axis.Regular(40, 0., 200., name='var', label=r'lead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    # # 'lepton1_pfIsoId': hist.axis.Integer(0, 12, name='var', label=r'$l_{lead}$ PF IsoId', growth=False, underflow=False, overflow=False), 
    # 'lepton1_pfIsoId': hist.axis.Regular(12, 0, 12, name='var', label=r'$l_{lead}$ PF IsoId', growth=False, underflow=False, overflow=False), 
    # 'lepton1_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\l_{lead}$ MVA ID', growth=False, underflow=False, overflow=False),  

    # # diphoton variables
    # 'pt': hist.axis.Regular(40, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=False, underflow=False, overflow=False), 

    # # angular (cos) variables
    # 'nonRes_CosThetaStar_CS': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{CS})$', growth=False, underflow=False, overflow=False), 
    # 'nonRes_CosThetaStar_jj': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{jj})$', growth=False, underflow=False, overflow=False), 
    # 'nonRes_CosThetaStar_gg': hist.axis.Regular(50, -1., 1., name='var', label=r'cos$(\theta_{gg})$', growth=False, underflow=False, overflow=False),

    # # photon variables
    # 'lead_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{lead}$ MVA ID', growth=False, underflow=False, overflow=False), 
    # 'lead_sigmaE_over_E': hist.axis.Regular(50, 0., 0.06, name='var', label=r'$\gamma_1 \sigma {E} / E$', growth=False, underflow=False, overflow=False), 
    # # ----------
    # 'sublead_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{sublead}$ MVA ID', growth=False, underflow=False, overflow=False),
    # 'sublead_sigmaE_over_E': hist.axis.Regular(50, 0., 0.06, name='var', label=r'$\gamma_2 \sigma {E} / E$', growth=False, underflow=False, overflow=False),

    # # dijet variables
    # 'dijet_PNetRegPt': hist.axis.Regular(100, 0., 500., name='var', label=r'jj $p_T$ [GeV]', growth=False, underflow=False, overflow=False),

    # # HH variables
    # 'HH_PNetRegPt': hist.axis.Regular(100, 0., 700., name='var', label=r'HH $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    # 'HH_PNetRegEta': hist.axis.Regular(50, -5., 5., name='var', label=r'HH $\eta$', growth=False, underflow=False, overflow=False),

    # # ATLAS variables #
    # 'RegPt_balance': hist.axis.Regular(100, 0., 2., name='var', label=r'$p_{T,HH} / (p_{T,\gamma1} + p_{T,\gamma2} + p_{T,j1} + p_{T,j2})$', growth=False, underflow=False, overflow=False), 
    
    # # VH variables #
    # 'DeltaPhi_jj': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,j_2)$', growth=False, underflow=False, overflow=False),
    # 'DeltaEta_jj': hist.axis.Regular(20, 0., 10., name='var', label=r'$\Delta\eta (j_1,j_2)$', growth=False, underflow=False, overflow=False),
    # 'isr_jet_RegPt': hist.axis.Regular(100, 0., 200., name='var', label=r'ISR jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'DeltaPhi_isr_jet_z': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_{ISR},jj)$', growth=False, underflow=False, overflow=False),

    # # BDT output #
    # 'MultiBDT_output': hist.axis.Regular(100, 0., 1., name='var', label=r'Multiclass BDT output', growth=False, underflow=False, overflow=False), 
}
BLINDED_VARIABLES = {
    # diphoton variables
    'mass': (
        hist.axis.Regular(55, 80., 180., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False),
        [115, 135]
    )
}
EXTRA_MC_VARIABLES = {
    'eventWeight', MC_DATA_MASK
}
EXTRA_DATA_VARIABLES = {
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

def sideband_cuts(sample):
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
    )
    sample[MC_DATA_MASK] = event_mask

    if 'eventWeight' in sample.fields: 
        sample['eventWeight'] = sample['eventWeight'] * (LUMINOSITIES['total_lumi'] / (LUMINOSITIES['total_lumi'] - LUMINOSITIES['2024']))

def get_mc_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    
    # Pull MC sample dir_list
    for sim_era in dir_lists.keys():
        # dir_lists[sim_era] = list(os.listdir(sim_era))
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
        {field: sample[field] for field in (set(VARIABLES.keys()) | set(BLINDED_VARIABLES.keys()) | extra_variables)}
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
    pq_dict: dict, variable: str, axis, blind_edges=None
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

def plot(
    variable: str, hists: dict, 
    era='2022postEE', lumi=0.0,
    rel_dirpath='', histtypes=None, density=False
):
    """
    Plots and saves out the data-MC comparison histogram
    """
    # Initiate figure
    fig, axs = plt.subplots(
        2, 1, sharex=True, height_ratios=[4,1], figsize=(10, 8)
    )
    linewidth=2.

    hist_names = list(hists.keys())
    if histtypes is None:
        histtypes = ["step" for _ in range(len(hists))]
    for idx, hist_name in enumerate(hist_names):
        hep.histplot(
            hists[hist_name], label=hist_name, 
            yerr=hists[hist_name].variances() if (APPLY_WEIGHTS and re.search('mc', hist_name.lower()) is not None) else True,
            ax=axs[0], lw=linewidth, histtype=histtypes[idx], alpha=0.8,
            density=density
        )
    
    # Plotting niceties #
    hep.cms.lumitext(f"{era} {lumi:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend()
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(hists[hist_names[0]].axes.label[0])
    axs[0].set_yscale('linear')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{variable}_{hist_names[0]}{"_"+hist_names[-1] if len(hist_names) > 1 else ""}.pdf')
    plt.savefig(f'{destdir}1dhist_{variable}_{hist_names[0]}{"_"+hist_names[-1] if len(hist_names) > 1 else ""}.png')
    plt.close()

def get_concat_samples(sample_dirs: dict, save=False):
    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    sample_pqs = {}
    for dir_name, dir_dict in sample_dirs.items():
        print('======================== \n', dir_name+" started")
        sample_pqs[dir_name] = {}

        for sample_era, sample_list in dir_dict.items():
            print('======================== \n', sample_era+" started")
            sample_pqs[dir_name][sample_era] = []

            for samplefilepath in sample_list:
                print('======================== \n', samplefilepath[:samplefilepath.rfind('.')]+" started")

                sample = ak.from_parquet(samplefilepath)
                sideband_cuts(sample)
                sample_pqs[dir_name][sample_era].append(
                    slimmed_parquet(
                        sample, 
                        EXTRA_MC_VARIABLES if re.search('mc', dir_name.lower()) is not None else EXTRA_DATA_VARIABLES
                    )
                )

                if save:
                    output_pq_filepath = (
                        samplefilepath[:samplefilepath.find('Run3_202')+len(lpc_filegroup('202x'))]
                        + '_Cut_output_'
                        + samplefilepath[samplefilepath.find('Run3_202')+len(lpc_filegroup('202x')):]
                    )
                    if not os.path.exists(output_pq_filepath[:output_pq_filepath.rfind('/')]):
                        os.makedirs(output_pq_filepath[:output_pq_filepath.rfind('/')])
                    ak.to_parquet(sample[MC_DATA_MASK], output_pq_filepath)
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
    era=None, lumi=None
):
    """
    Performs the sample comparison.
    """

    for dir_name, dir_dict in sample_dirs.items():
        if re.search('mc', dir_name.lower()) is not None:
            get_mc_dir_lists(dir_dict)
        elif re.search('data', dir_name.lower()) is not None:
            get_data_dir_lists(dir_dict)
        else:
            raise Exception(f"Ambiguous whether dirlist {dir_name} is Data or MC, please include either word in the name (key) of the dirlist.")
        
    concat_samples = get_concat_samples(sample_dirs, save=save)

    plot_list = []
    if plottype == 'comparison':
        plot_list.append(concat_samples)
    elif plottype == 'split':
        for dir_name, dir_dict in concat_samples:
            plot_list.append({dir_name: dir_dict})
    elif plottype == 'Data/MC':
        simplified_concat = {'Data': {}, 'MC': {}}
        for dir_name, dir_dict in concat_samples:
            if re.search('data', dir_name.lower()) is not None:
                simplified_concat['Data'][dir_name] = dir_dict
            else:
                simplified_concat['MC'][dir_name] = dir_dict

    for concat_dict in plot_list:
        # Ploting over variables for MC and Data
        for variable, axis in VARIABLES.items():
            hists = generate_hists(
                concat_dict, variable, axis
            )
            plot(
                variable, hists, era=era, lumi=lumi, 
                density=density, datamc=(plottype == 'Data/MC')  # need to implement
            )

        # # Ploting over variables for MC and Data
        for variable, (axis, blind_edges) in BLINDED_VARIABLES.items():
            hists = generate_hists(
                concat_dict, variable, axis,
                blind_edges=blind_edges
            )
            plot(
                variable, hists, era=era, lumi=lumi, 
                density=density, datamc=(plottype == 'Data/MC')
            )
            

if __name__ == '__main__':
    sample_dirs = {
        'Data-2022-24': {
            os.path.join(LPC_FILEPREFIX_22[:-len('sim/')], "data", ""): None,
            os.path.join(LPC_FILEPREFIX_23[:-len('sim/')], "data", ""): None,
            os.path.join(LPC_FILEPREFIX_24[:-len('sim/')], "data", ""): None
        },
        'MC-2022-24-GluGluToHH': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GluGluToHH", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "preBPix", "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
            os.path.join(LPC_FILEPREFIX_23, "postBPix", "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00", ""): None,
        },
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
        'MC-2022-24-GGJets': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GGJets", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GGJets", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "preBPix", "GGJets", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postBPix", "GGJets", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GGJets", ""): None,
        },
        'MC-2022-24-GJetPt20To40': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GJetPt20To40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GJetPt20To40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "preBPix", "GJetPt20To40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postBPix", "GJetPt20To40", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GJetPt20To40", ""): None,
        },
        'MC-2022-24-GJetPt40': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", "GJetPt40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postEE", "GJetPt40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "preBPix", "GJetPt40", ""): None,
            os.path.join(LPC_FILEPREFIX_22, "postBPix", "GJetPt40", ""): None,
            # os.path.join(LPC_FILEPREFIX_24, "GJetPt40", ""): None,
        },
        # 'MC-2022-24-VBFToHH': {
        #     os.path.join(LPC_FILEPREFIX_22, "preEE", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_22, "postEE", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "preBPix", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        #     os.path.join(LPC_FILEPREFIX_23, "postBPix", "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        #     # os.path.join(LPC_FILEPREFIX_24, "VBFHHto2B2G_CV_1_C2V_1_C3_1", ""): None,
        # }
    }

    main(
        sample_dirs, density=False,
        era="2022-24", lumi=LUMINOSITIES["total_lumi"],
    )