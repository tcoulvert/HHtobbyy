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
lpc_filegroup = lambda s: f'Run3_{s}_merged'
# lpc_filegroup = lambda s: f'Run3_{s}_merged_MultiBDT_output_mvaIDCorr_22_23'
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, lpc_filegroup('2022'), 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, lpc_filegroup('2023'), 'sim', '')
LPC_FILEPREFIX_24 = os.path.join(lpc_fileprefix, lpc_filegroup('2024'), 'sim', '')
END_FILEPATH = '*output.parquet' if re.search('MultiBDT_output', LPC_FILEPREFIX_22) is not None else '*merged.parquet'

DESTDIR = 'sample_check_plots'
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
    os.path.join(LPC_FILEPREFIX_22, "preEE", ""): 7.9804,
    os.path.join(LPC_FILEPREFIX_22, "postEE", ""): 26.6717,
    os.path.join(LPC_FILEPREFIX_23, "preBPix", ""): 17.794,
    os.path.join(LPC_FILEPREFIX_23, "postBPix", ""): 9.451,
    os.path.join(LPC_FILEPREFIX_24, ""): 109.08,
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

def sideband_cuts(sample):
    """
    Builds the event_mask used to do data-mc comparison in a sideband.
    """
    # Require diphoton and dijet exist (required in preselection, and thus is all True)
    event_mask = (
        sample['is_nonRes']
        & (
            sample['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in sample.fields else sample['pass_fiducial_geometric']
        )
    )
    sample[MC_DATA_MASK] = event_mask

def check_variables(sample, only_bare=False):
    """
    Checks the variables desired and prints out info 
    -> (just a place holder to make the code clean)
    """
    jer_dict = {
        # -5.191 < eta < -3.139
        (-5.191, -3.139): {
            # 10 < jet pt [GeV] < 121
            (10, 121): [1.124, 0.990, 1.258],  # [SF, SF_down, SF_up]
            (121, 140): [1.124, 0.990, 1.258],
            (150, 191): [1.184, 1.100, 1.268],
            (249, 272): [1.261, 1.173, 1.348],
        },
        (-3.139, -2.964): {
            (10, 121): [1.206, 1.088, 1.323],
            (121, 140): [1.206, 1.088, 1.323],
            (150, 192): [1.242, 1.161, 1.323],
            (249, 272): [1.260, 1.193, 1.328],
        },
        (-2.500, -2.322): {
            (10, 128): [1.160, 1.094, 1.225],
            (128, 145): [1.160, 1.094, 1.225],
            (177, 195): [1.160, 1.100, 1.220],
            (246, 283): [1.160, 1.106, 1.215],
        },
        (-1.740, -1.566): {
            (10, 128): [1.203, 1.154, 1.252],
            (128, 145): [1.203, 1.154, 1.252],
            (177, 195): [1.204, 1.163, 1.245],
            (246, 284): [1.211, 1.179, 1.242],
        },
        (-1.305, -1.044): {
            (10, 128): [1.055, 1.033, 1.077],
            (128, 145): [1.055, 1.033, 1.077],
            (177, 195): [1.044, 1.025, 1.063],
            (246, 284): [1.034, 1.018, 1.050],
        },
        (-0.783, -0.522): {
            (10, 128): [1.109, 1.094, 1.124],
            (128, 145): [1.109, 1.094, 1.124],
            (177, 195): [1.096, 1.083, 1.109],
            (246, 284): [1.083, 1.072, 1.094],
        },
        (-0.261, -0.000): {
            (10, 128): [1.128, 1.110, 1.146],
            (128, 145): [1.128, 1.110, 1.146],
            (177, 195): [1.114, 1.098, 1.130],
            (246, 284): [1.100, 1.086, 1.113],
        },
        (0.000, 0.261): {
            (10, 128): [1.128, 1.110, 1.146],
            (128, 145): [1.128, 1.110, 1.146],
            (177, 195): [1.114, 1.098, 1.130],
            (246, 284): [1.100, 1.086, 1.113],
        },
        (0.522, 0.783): {
            (10, 128): [1.109, 1.094, 1.124],
            (128, 145): [1.109, 1.094, 1.124],
            (177, 195): [1.096, 1.083, 1.109],
            (246, 284): [1.083, 1.072, 1.094],
        },
        (1.044, 1.305): {
            (10, 128): [1.055, 1.033, 1.077],
            (128, 145): [1.055, 1.033, 1.077],
            (177, 195): [1.044, 1.025, 1.063],
            (246, 284): [1.034, 1.018, 1.050],
        },
        (1.566, 1.740): {
            (10, 128): [1.203, 1.154, 1.252],
            (128, 145): [1.203, 1.154, 1.252],
            (177, 195): [1.204, 1.163, 1.245],
            (246, 284): [1.211, 1.179, 1.242],
        },
        (2.322, 2.500): {
            (10, 128): [1.160, 1.094, 1.225],
            (128, 145): [1.160, 1.094, 1.225],
            (177, 195): [1.160, 1.100, 1.220],
            (246, 283): [1.160, 1.106, 1.215],
        },
        (2.964, 3.139): {
            (10, 121): [1.206, 1.088, 1.323],
            (121, 140): [1.206, 1.088, 1.323],
            (150, 192): [1.242, 1.161, 1.323],
            (249, 272): [1.260, 1.193, 1.328],
        },
        (3.139, 5.191): {
            (10, 121): [1.124, 0.990, 1.258],
            (121, 140): [1.124, 0.990, 1.258],
            (150, 191): [1.184, 1.100, 1.268],
            (249, 272): [1.261, 1.173, 1.348],
        },
    }

    for field in sample.fields:
        if re.search('jet1', field) is not None:
            print(field)
            print('-'*60)

    jet_pts = np.array(
        [ak.to_numpy(sample[f"jet{jet_idx}_pt"], allow_missing=False)
        for jet_idx in range(1, 11)]
    )
    jet_etas = np.array(
       [ak.to_numpy(sample[f"jet{jet_idx}_eta"], allow_missing=False)
        for jet_idx in range(1, 11)]
    )
    eventIDs = np.array(
        [ak.to_numpy(sample[f"event"], allow_missing=False)
        for jet_idx in range(1, 11)]
    )
    jetIdxs = np.array(
        [jet_idx*np.ones_like(jet_pts[0])
        for jet_idx in range(1, 11)]
    )

    # def smear_pt_fctr(jet_pt, smear_fctr, seed=21):
    #     rng = np.random.default_rng(seed=seed)

    #     norm = rng.normal(
    #         loc=np.zeros_like(jet_pt), scale=
    #     )

    for eta_range in jer_dict.keys():
        eta_mask = np.logical_and(jet_etas >= eta_range[0], jet_etas < eta_range[1])
        for pt_range in jer_dict[eta_range].keys():
            pt_mask = np.logical_and(jet_pts >= pt_range[0], jet_pts < pt_range[1])
            
            mask = np.logical_and(eta_mask, pt_mask)

            smear_pt = lambda jet_pt, smear_fctr: np.random.normal(
                loc=jet_pt, scale=jet_pt * np.sqrt(np.abs((smear_fctr**2) - 1)),
            )
            
            # if selected_jet_pt is None:
            #     print(f"Could not find suitable jet for eta range {eta_range} and pt range {pt_range}")
            #     print('='*60)
            #     continue
            # print(f"eta range {eta_range}, pt range {pt_range}")
            # print('-'*60)
            # print(f"eventID = {selected_eventid}")
            # print(f"jer NOM jet pt = {selected_jet_pt:.5f}")
            # if not only_bare:
            #     # def smear_pt(jet_pt, smear_ftr):
            #     #     while True:
            #     #         rand_sample = np.random.normal(
            #     #             loc=jet_pt, scale=jet_pt * np.sqrt(np.abs((smear_ftr**2) - 1)),
            #     #             # size=100
            #     #         )
            #     #         if (
            #     #             rand_sample > jet_pt - (jet_pt * np.sqrt(np.abs((smear_ftr**2) - 1)))
            #     #             and rand_sample < jet_pt + (jet_pt * np.sqrt(np.abs((smear_ftr**2) - 1)))
            #     #         ): return rand_sample
            #     # # print(f"jer NOM jet pt = {smear_pt(selected_jet_pt, jer_dict[eta_range][pt_range][0]):.5f}")
            #     # print(f"jer UP jet pt = {smear_pt(selected_jet_pt, jer_dict[eta_range][pt_range][2]):.5f}")
            #     # print(f"jer DOWN jet pt = {smear_pt(selected_jet_pt, jer_dict[eta_range][pt_range][1]):.5f}")
            #     def smear_pt_bounds(jet_pt, smear_ftr):
            #         return (
            #             jet_pt - (jet_pt * np.sqrt(np.abs((smear_ftr**2) - 1))),
            #             jet_pt + (jet_pt * np.sqrt(np.abs((smear_ftr**2) - 1)))
            #         )
            #     print(f"jer NOM jet pt 1sigma bounds = {smear_pt_bounds(selected_jet_pt, jer_dict[eta_range][pt_range][0])}")
            #     print(f"jer UP jet pt 1sigma bounds = {smear_pt_bounds(selected_jet_pt, jer_dict[eta_range][pt_range][2])}")
            #     print(f"jer DOWN jet pt 1sigma bounds = {smear_pt_bounds(selected_jet_pt, jer_dict[eta_range][pt_range][1])}")
            # print('='*60)

def get_mc_dir_lists(dir_lists: dict):
    """
    Builds the dictionary of lists of samples to use in comparison.
      -> Automatically checks if the merger.py file has been run.
    """
    
    # Pull MC sample dir_list
    for sim_era in dir_lists.keys():
        dir_lists[sim_era] = list(os.listdir(sim_era))
        dir_lists[sim_era].sort()

def get_data_dir_lists(dir_lists: dict):
    
    # Pull Data sample dir_list
    for data_era in dir_lists.keys():
        dir_lists[data_era] = list(os.listdir(data_era))
        dir_lists[data_era].sort()

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
        # # ggf H
        # 'GluGluHToGG': 'GluGluHToGG',
        # 'GluGluHToGG_M_125': 'GluGluHToGG',
        # 'GluGluHtoGG': 'GluGluHToGG',
        # # ttH
        # 'ttHToGG': 'ttHToGG',
        # 'ttHtoGG_M_125': 'ttHToGG',
        # 'ttHtoGG': 'ttHToGG',
        # # vbf H
        # 'VBFHToGG': 'VBFHToGG',
        # 'VBFHToGG_M_125': 'VBFHToGG',
        # 'VBFHtoGG': 'VBFHToGG',
        # # VH
        # 'VHToGG': 'VHToGG',
        # 'VHtoGG_M_125': 'VHToGG',
        # 'VHtoGG': 'VHToGG',
        # 'VHtoGG_M-125': 'VHToGG',
        # # bbH
        # 'BBHto2G_M_125': 'bbHToGG',
        # 'bbHtoGG': 'bbHToGG',
    }
    if dir_name in sample_name_map:
        return sample_name_map[dir_name]
    else:
        return None

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

def generate_hists(pq_dict: dict, variable: str, axis, density=False, blind_edges=None):
    # https://indico.cern.ch/event/1433936/ #
    # Generate syst hists and ratio hists
    hists = {}
    for ak_name, ak_arr in pq_dict.items():
        ak_hist = ak_arr[:, 0] if 'MultiBDT_output' in VARIABLES else ak_arr

        if blind_edges is not None:
            mask = (
                (
                    (ak_hist[variable] < blind_edges[0]) 
                    | (ak_hist[variable] > blind_edges[1])
                ) & (ak_hist[MC_DATA_MASK])
            )
        else:
            mask = ak_hist[MC_DATA_MASK]

        if re.search('mc', ak_name.lower()) and APPLY_WEIGHTS:
            hists[ak_name] = hist.Hist(axis, storage='weight').fill(
                var=ak_hist[variable][mask],
                weight=ak_hist['eventWeight'][mask],
            )
        else:
            hists[ak_name] = hist.Hist(axis).fill(
                var=ak_hist[variable][mask]
            )

    return hists

def plot(
    variable: str, hists: dict, 
    year='2022', era='postEE', lumi=0.0,
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
    hep.cms.lumitext(f"{year}{era} {lumi:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=axs[0])
    hep.cms.text("Work in Progress", ax=axs[0])
    # Plot legend properly
    axs[0].legend()
    # Push the stack and ratio plots closer together
    fig.subplots_adjust(hspace=0.05)
    # Plot x_axis label properly
    axs[0].set_xlabel('')
    axs[1].set_xlabel(hist_names[0]+'  '+hists[hist_names[0]].axes.label[0])
    axs[0].set_yscale('log') if variable == 'MultiBDT_output' else axs[0].set_yscale('linear')
    # Save out the plot
    destdir = os.path.join(DESTDIR, rel_dirpath, '')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    plt.savefig(f'{destdir}1dhist_{variable}_{hist_names[0]}{"_"+hist_names[-1] if len(hist_names) > 1 else ""}.pdf')
    plt.savefig(f'{destdir}1dhist_{variable}_{hist_names[0]}{"_"+hist_names[-1] if len(hist_names) > 1 else ""}.png')
    plt.close()
    
def main(
    sample_dirs, 
    year='2022', era='preEE',
    lumi=LUMINOSITIES[os.path.join(LPC_FILEPREFIX_22, "preEE", "")],
    density=False
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
        
    # Make parquet dicts, merged by samples and pre-slimmed (keeping only VARIABLES and EXTRA_VARIABLES)
    sample_pqs = {}
    for dir_name, dir_dict in sample_dirs.items():
        sample_pqs[dir_name] = {}

        for sample_era, sample_list in dir_dict.items():
            print('======================== \n', year+''+era+" started")
            sample_pqs[dir_name][sample_era] = []

            for sample_name in sample_list:
                if re.search('mc', dir_name.lower()) is not None:
                    std_samplename = find_dirname(sample_name)
                    if std_samplename is None:
                        print(f'{sample_name} not in samples selected for this computation.')
                        continue
                    if len(os.listdir(os.path.join(sample_era, sample_name))) == 1:
                        print(f'{sample_name} does not have variations computed.')
                        continue

                    sample_dirpath = os.path.join(sample_era, sample_name, 'nominal', END_FILEPATH)
                    sample_jerUp_dirpath = os.path.join(sample_era, sample_name, 'jer_syst_up', END_FILEPATH)
                    sample_jerDown_dirpath = os.path.join(sample_era, sample_name, 'jer_syst_down', END_FILEPATH)
                else: 
                    std_samplename = sample_name
                    sample_dirpath = os.path.join(sample_era, sample_name, END_FILEPATH)
                print('======================== \n', std_samplename+" started")

                sample = ak.from_parquet(glob.glob(sample_dirpath)[0])
                sideband_cuts(sample)

                check_variables(sample)

                if re.search('mc', dir_name.lower()) is not None:
                    sample_jerUp = ak.from_parquet(glob.glob(sample_jerUp_dirpath)[0])
                    sideband_cuts(sample_jerUp)
                    print('*'*100)
                    print('JER up')
                    print('*'*100)
                    check_variables(sample_jerUp, only_bare=True)

                    sample_jerDown = ak.from_parquet(glob.glob(sample_jerDown_dirpath)[0])
                    sideband_cuts(sample_jerDown)
                    print('*'*100)
                    print('JER down')
                    print('*'*100)
                    check_variables(sample_jerDown, only_bare=True)


                sample_pqs[dir_name][sample_era].append(
                    slimmed_parquet(
                        sample, 
                        EXTRA_MC_VARIABLES if re.search('mc', dir_name.lower()) is not None else EXTRA_DATA_VARIABLES
                    )
                )

                del sample
                print('======================== \n', std_samplename+" finished")

            print('======================== \n', year+''+era+" finished")

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


    # Ploting over variables for MC and Data
    for variable, axis in VARIABLES.items():
        hists = generate_hists(
            concat_samples, variable, axis,
            density=density
        )
        plot(
            variable, hists, 
            year=year, era=era, lumi=lumi, 
            density=density
        )

    # # Ploting over variables for MC and Data
    for variable, (axis, blind_edges) in BLINDED_VARIABLES.items():
        hists = generate_hists(
            concat_samples, variable, axis,
            blind_edges=blind_edges,
            density=density
        )
        plot(
            variable, hists, 
            year=year, era=era, lumi=lumi, 
            density=density
        )
            

if __name__ == '__main__':
    sample_dirs = {
        'MC-2022preEE': {
            os.path.join(LPC_FILEPREFIX_22, "preEE", ""): None,
        },
    }
    main(
        sample_dirs, 
        year='2022', era='preEE', lumi=LUMINOSITIES[os.path.join(LPC_FILEPREFIX_22, "preEE", "")], 
        density=False
    )