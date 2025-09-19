# %matplotlib widget
# Stdlib packages
import copy
import datetime
import glob
import json
import os
import re
import warnings
from pathlib import Path

# Common Py packages
import awkward as ak
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.special import logit as inverse_sigmoid

# HEP packages
import gpustat
import h5py
import hist
import mplhep as hep
import xgboost as xgb
from cycler import cycler

# ML packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
from sklearn.metrics import log_loss
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Module packages
from plotting_utils import (
    plot_filepath, 
    get_ttH_score, get_QCD_score,
    cat_mask
)

################################


gpustat.print_gpustat()

plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

################################


# Dictionary of variables
VARIABLES = {
    # key: hist.axis axes for plotting #
    # MET variables
    'puppiMET_sumEt': hist.axis.Regular(40, 150., 2000, name='var', label=r'puppiMET $\Sigma E_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'puppiMET $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'puppiMET_phi': hist.axis.Regular(20,-3.2, 3.2, name='var', label=r'puppiMET $\phi$', growth=False, underflow=False, overflow=False), 
    # jet-MET variables
    'DeltaPhi_j1MET': hist.axis.Regular(20,-3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    'DeltaPhi_j2MET': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_2,E_T^{miss})$', growth=False, underflow=False, overflow=False), 
    # jet-photon variables
    'DeltaR_jg_min': hist.axis.Regular(30, 0, 5, name='var', label=r'min$(\Delta R(jet, \gamma))$', growth=False, underflow=False, overflow=False), 
    # jet variables
    # 'jet1_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    # 'jet2_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'sublead jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'lead_bjet_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'lead_bjet_eta': hist.axis.Regular(20, -5., 5., name='var', label=r'lead bjet $\eta$', growth=False, underflow=False, overflow=False),
    'nonRes_lead_bjet_btagPNetB': hist.axis.Regular(50, 0., 1., name='var', label=r'$j_{lead}$ PNet btag score', growth=False, underflow=False, overflow=False), 
    'sublead_bjet_pt': hist.axis.Regular(40, 20., 250, name='var', label=r'lead bjet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'sublead_bjet_eta': hist.axis.Regular(20, -5., 5., name='var', label=r'lead bjet $\eta$', growth=False, underflow=False, overflow=False),
    'nonRes_sublead_bjet_btagPNetB': hist.axis.Regular(50, 0., 1., name='var', label=r'$j_{sublead}$ PNet btag score', growth=False, underflow=False, overflow=False),
    'n_jets': hist.axis.Integer(0, 10, name='var', label=r'$n_{jets}$', growth=False, underflow=False, overflow=False), 
    'chi_t0': hist.axis.Regular(40, 0., 150, name='var', label=r'$\chi_{t0}^2$', growth=False, underflow=False, overflow=False), 
    'chi_t1': hist.axis.Regular(30, 0., 500, name='var', label=r'$\chi_{t1}^2$', growth=False, underflow=False, overflow=False), 
    # lepton variables
    'n_leptons': hist.axis.Integer(0, 10, name='var', label=r'$n_{leptons}$', growth=False, underflow=False, overflow=False),
    'lepton1_pt': hist.axis.Regular(40, 0., 200., name='var', label=r'lead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton2_pt': hist.axis.Regular(40, 0., 200., name='var', label=r'sublead lepton $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'lepton1_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'lead lepton $\eta$', growth=False, underflow=False, overflow=False), 
    'lepton2_eta': hist.axis.Regular(30, -5., 5., name='var', label=r'sublead lepton $\eta$', growth=False, underflow=False, overflow=False),
    'lepton1_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'lead lepton $\phi$', growth=False, underflow=False, overflow=False), 
    'lepton2_phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'sublead lepton $\phi$', growth=False, underflow=False, overflow=False),
    # diphoton variables
    'pt': hist.axis.Regular(40, 20., 2000, name='var', label=r' $\gamma\gamma p_{T}$ [GeV]', growth=False, underflow=False, overflow=False),
    'eta': hist.axis.Regular(20, -5., 5., name='var', label=r'$\gamma\gamma \eta$', growth=False, underflow=False, overflow=False), 
    'phi': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\gamma \gamma \phi$', growth=False, underflow=False, overflow=False),
    # angular (cos) variables
    'nonRes_CosThetaStar_CS': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{CS})$', growth=False, underflow=False, overflow=False), 
    'nonRes_CosThetaStar_jj': hist.axis.Regular(20, -1, 1, name='var', label=r'cos$(\theta_{jj})$', growth=False, underflow=False, overflow=False), 
    'nonRes_CosThetaStar_gg': hist.axis.Regular(50, -1., 1., name='var', label=r'cos$(\theta_{gg})$', growth=False, underflow=False, overflow=False),
    # jet-lepton variables
    'leadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'leadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{lead}, l_{sublead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_leadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{lead})$', growth=False, underflow=False, overflow=False), 
    'subleadBjet_subleadLepton': hist.axis.Regular(30, 0, 5, name='var', label=r'$\Delta R(bjet_{sublead}, l_{sublead})$', growth=False, underflow=False, overflow=False),
    # Yibo's BDT variables
    'lead_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{lead}$ MVA ID', growth=False, underflow=False, overflow=False), 
    'sublead_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{sublead}$ MVA ID', growth=False, underflow=False, overflow=False),
    'lead_sigmaE_over_E': hist.axis.Regular(50, 0., 0.06, name='var', label=r'$\gamma_1 \sigma {E} / E$', growth=False, underflow=False, overflow=False), 
    'sublead_sigmaE_over_E': hist.axis.Regular(50, 0., 0.06, name='var', label=r'$\gamma_2 \sigma {E} / E$', growth=False, underflow=False, overflow=False),
    'lead_bjet_pt_over_Mjj': hist.axis.Regular(50, 0., 4., name='var', label=r'$j1 p_{T} / M_{jj}$', growth=False, underflow=False, overflow=False), 
    'sublead_bjet_pt_over_Mjj': hist.axis.Regular(50, 0., 2., name='var', label=r'$j2 p_{T} / M_{jj}$', growth=False, underflow=False, overflow=False),
    'lead_bjet_sigmapT_over_pT': hist.axis.Regular(50, 0., 0.02, name='var', label=r'$j1 \sigma p_{T} / p_{T}$', growth=False, underflow=False, overflow=False), 
    'sublead_bjet_sigmapT_over_pT': hist.axis.Regular(50, 0., 0.02, name='var', label=r'$j2 \sigma p_{T} / p_{T}$', growth=False, underflow=False, overflow=False),
    'dipho_mass_over_Mggjj': hist.axis.Regular(50, 0., 1., name='var', label=r'$M_{\gamma\gamma} / M_{\gamma\gamma jj}$', growth=False, underflow=False, overflow=False), 
    'dijet_mass_over_Mggjj': hist.axis.Regular(50, 0., 1., name='var', label=r'$M_{jj} / M_{\gamma\gamma jj}$', growth=False, underflow=False, overflow=False),
    # My variables for non-reso reduction #
    'lead_pfRelIso03_all_quadratic': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{lead}$ PF RelIso03 all quad.', growth=False, underflow=False, overflow=False), 
    'sublead_pfRelIso03_all_quadratic': hist.axis.Regular(50, -1., 1., name='var', label=r'$\gamma_{sublead}$ PF RelIso03 all quad.', growth=False, underflow=False, overflow=False),
    # Michael's DNN variables #
    'DeltaR_j1g1': hist.axis.Regular(50, 0., 5., name='var', label=r'$\Delta R(bjet_{lead}, \gamma_{lead})$', growth=False, underflow=False, overflow=False), 
    'DeltaR_j1g2': hist.axis.Regular(50, 0., 5., name='var', label=r'$\Delta R(bjet_{lead}, \gamma_{sublead})$', growth=False, underflow=False, overflow=False), 
    'DeltaR_j2g1': hist.axis.Regular(50, 0., 5., name='var', label=r'$\Delta R(bjet_{sublead}, \gamma_{lead})$', growth=False, underflow=False, overflow=False), 
    'DeltaR_j2g2': hist.axis.Regular(50, 0., 5., name='var', label=r'$\Delta R(bjet_{sublead}, \gamma_{sublead})$', growth=False, underflow=False, overflow=False),
    'HHbbggCandidate_pt': hist.axis.Regular(100, 0., 700., name='var', label=r'HH $p_T$ [GeV]', growth=False, underflow=False, overflow=False), 
    'HHbbggCandidate_eta': hist.axis.Regular(50, -5., 5., name='var', label=r'HH $\eta$', growth=False, underflow=False, overflow=False), 
    'HHbbggCandidate_phi': hist.axis.Regular(50, -3.2, 3.2, name='var', label=r'HH $\phi$', growth=False, underflow=False, overflow=False),
    'HHbbggCandidate_mass': hist.axis.Regular(25, 0., 700., name='var', label=r'$M_{HH}$ [GeV]', growth=False, underflow=False, overflow=False),
    # ATLAS variables #
    'pt_balance': hist.axis.Regular(100, 0., 2., name='var', label=r'$p_{T,HH} / (p_{T,\gamma1} + p_{T,\gamma2} + p_{T,j1} + p_{T,j2})$', growth=False, underflow=False, overflow=False), 
    # VH variables #
    'DeltaPhi_jj': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_1,j_2)$', growth=False, underflow=False, overflow=False),
    'DeltaEta_jj': hist.axis.Regular(20, 0., 10., name='var', label=r'$\Delta\eta (j_1,j_2)$', growth=False, underflow=False, overflow=False),
    'isr_jet_pt': hist.axis.Regular(100, 0., 200., name='var', label=r'ISR jet $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'DeltaPhi_isr_jet_z': hist.axis.Regular(20, -3.2, 3.2, name='var', label=r'$\Delta\phi (j_{ISR},jj)$', growth=False, underflow=False, overflow=False),
    'dijet_pt': hist.axis.Regular(100, 0., 500., name='var', label=r'jj $p_T$ [GeV]', growth=False, underflow=False, overflow=False),
    'lepton1_pfIsoId': hist.axis.Integer(0, 12, name='var', label=r'$l_{lead}$ PF IsoId', growth=False, underflow=False, overflow=False), 
    'lepton1_mvaID': hist.axis.Regular(50, -1., 1., name='var', label=r'$\l_{lead}$ MVA ID', growth=False, underflow=False, overflow=False), 

    # aux variables #
    'dijet_mass': hist.axis.Regular(55, 70., 180., name='var', label=r'$M_{jj}$ [GeV]', growth=False, underflow=False, overflow=False),
    'mass': hist.axis.Regular(55, 70., 180., name='var', label=r'$M_{\gamma\gamma}$ [GeV]', growth=False, underflow=False, overflow=False),
}

def make_input_plot(
    output_dir, var_name, hist_list, fold_idx=None, labels=None, density=True, 
    plot_prefix='', plot_postfix='', alpha=0.7, num_compare=1, log=False
):
    fig, ax = plt.subplots()
    
    if num_compare > 1:
        unique_linestyles = ["solid", "dashed", "dotted", "dashdot"]
        linestyles = unique_linestyles[:num_compare]
        linestyles = linestyles * ((len(hist_list) // len(linestyles)) + 1)
        linestyles = linestyles[:len(hist_list)]
        
        colors = [cmap for cmap in cmap_petroff10 for _ in range(0, num_compare)]
        colors = colors[:len(hist_list)]
        hep.histplot(
            hist_list, ax=ax, linewidth=3, histtype="step", yerr=True, density=density,
            linestyle=linestyles, label=labels, alpha=alpha, color=colors
        )
    else:
        hep.histplot(
            hist_list, ax=ax, linewidth=3, histtype="step", yerr=True, density=density,
            label=labels, alpha=alpha
        )

    # Plotting niceties #
    hep.cms.lumitext(f"{LUMINOSITIES['total_lumi']:.2f}" + r"fb$^{-1}$ (13.6 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)
    # Plot legend properly
    ax.legend(bbox_to_anchor=(1, 0.5))

    if log:
        ax.set_yscale('log')

    if fold_idx is not None:
        plt.savefig(f'{output_dir}/{plot_prefix}1dhist_{var_name}{plot_postfix}_fold{fold_idx}.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/{plot_prefix}1dhist_{var_name}{plot_postfix}_fold{fold_idx}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/{plot_prefix}1dhist_{var_name}{plot_postfix}.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/{plot_prefix}1dhist_{var_name}{plot_postfix}.png', bbox_inches='tight')

    plt.close()

