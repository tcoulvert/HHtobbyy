# Common Py packages
import numpy as np  
import pandas as pd

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import match_sample

################################

resolved_bTagWPs = {
    # MC
    '2016*preVFP': ("btagUParTAK4B", {'L': 0.0387, 'M': 0.1847, 'T': 0.5467, 'XT': 0.6777, 'XXT': 0.9218}),
    '2016*postVFP': ("btagUParTAK4B", {'L': 0.0400, 'M': 0.1898, 'T': 0.5538, 'XT': 0.6872, 'XXT': 0.9353}),
    '2017': ("btagUParTAK4B", {'L': 0.0331, 'M': 0.1776, 'T': 0.5755, 'XT': 0.7274, 'XXT': 0.9666}),
    '2018': ("btagUParTAK4B", {'L': 0.0308, 'M': 0.1610, 'T': 0.5405, 'XT': 0.6992, 'XXT': 0.9655}),

    '2022*(preEE)|((Data|Era)[CD])': ("btagPNetB", {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961, '3XT': 0.9986, '4XT': 0.999}),
    '2022*(postEE)|((Data|Era)[EFG])': ("btagPNetB", {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664, '3XT': 0.9986, '4XT': 0.999}),
    '2023*(preBPix)|((Data|Era)[C])': ("btagPNetB", {'L': 0.0358, 'M': 0.1917, 'T': 0.6172, 'XT': 0.7515, 'XXT': 0.9659, '3XT': 0.9986, '4XT': 0.999}),
    '2023*(postBPix)|((Data|Era)[D])': ("btagPNetB", {'L': 0.0359, 'M': 0.1919, 'T': 0.6133, 'XT': 0.7544, 'XXT': 0.9688, '3XT': 0.9986, '4XT': 0.999}),
    '2024': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739, '3XT': 0.9983, '4XT': 0.9987}),  # 3XT was calculated to have ggF HH kl-1p00 lead *OR* sublead bjets pass with 25% efficiency, 4XT calculated for 10% efficiency
    '2025': ("btagUParTAK4B", {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739, '3XT': 0.9983, '4XT': 0.9987}), 
}

NUM_JETS = 10

################################


# Variables to add for resolved training
def add_bTagWP_resolved(df: pd.DataFrame, filepath: str, prefactor: str):
    bTagVar, WP_dict = resolved_bTagWPs[match_sample(filepath, resolved_bTagWPs.keys())]
    
    for bjet_type in ['lead', 'sublead']:
        for i, (WPname, WP) in enumerate(WP_dict.items()):
            df[f"{prefactor}_{bjet_type}_bjet_bTagWP{WPname}"] = np.where(df[f"{prefactor}_{bjet_type}_bjet_{bTagVar}"] > WP, 1, 0)
            if WPname not in ['L', 'M', 'T', 'XT', 'XXT']: continue
            elif i == 0: df[f"{prefactor}_{bjet_type}_bjet_bTagWP"] = np.zeros(len(df))
            df[f"{prefactor}_{bjet_type}_bjet_bTagWP"] = np.where(df[f"{prefactor}_{bjet_type}_bjet_{bTagVar}"] > WP, i+1, df[f"{prefactor}_{bjet_type}_bjet_bTagWP"])

def add_vars_resolvedMLP(df: pd.DataFrame, filepath: str, prefactor: str):
    add_bTagWP_resolved(df, filepath, prefactor=prefactor)

    ### BEGIN Manos variables ###
    df[f"{prefactor}_diphoton_PtOverM_ggjj"] = df["pt"] / df[f"{prefactor}_HHbbggCandidate_mass"]
    df[f"{prefactor}_dijet_PtOverM_ggjj"] = df[f"{prefactor}_dijet_pt"] / df[f"{prefactor}_HHbbggCandidate_mass"]

    df[f"{prefactor}_lead_bjet_over_M_regressed"] = df[f"{prefactor}_lead_bjet_pt"] / df[f"{prefactor}_dijet_mass_DNNreg"]
    df[f"{prefactor}_sublead_bjet_over_M_regressed"] = df[f"{prefactor}_sublead_bjet_pt"] / df[f"{prefactor}_dijet_mass_DNNreg"]
    ### END Manos variables ###

    df['pass_presel'] = (
        (df['lead_mvaID'] > -0.7) & (df['sublead_mvaID'] > -0.7)
        & (df[f"{prefactor}_dijet_mass_DNNreg"] > 80) & (df[f"{prefactor}_dijet_mass_DNNreg"] < 190)
    )
