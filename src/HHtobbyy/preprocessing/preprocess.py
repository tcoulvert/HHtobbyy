# Common Py packages
import numpy as np
import pandas as pd

from HHtobbyy.preprocessing.preprocessing_utils import (
    match_sample_name, match_sample_xs, match_sample_lumi, match_sample_era
)
from HHtobbyy.workspace_utils.retrieval_utils import match_sample

################################


# MC Era: total era luminosity [fb^-1] #
luminosities = {
    '2016*preVFP': 19.5,
    '2016*postVFP': 16.8, 
    '2017': 42.07, 
    '2018': 59.56,
    '2022*preEE': 7.9804,
    '2022*postEE': 26.6717,
    '2023*preBPix': 17.794,
    '2023*postBPix': 9.451,
    '2024': 109.08,
    '2025': 110.73 
}
# Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
xs_name_map = {
    ## Signal ##
    'Run2*GluGluToHH*kl.1p00*kt.1p00*c2.0p00': ('GluGluToHH_kl1p00_kt1p00_c20p00', 8.1e-02), 
    'Run3*GluGluToHH*kl.1p00*kt.1p00*c2.0p00': ('GluGluToHH_kl1p00_kt1p00_c20p00', 34.43*0.0026), 

    'Run2*GluGluToHH*kl.5p00*kt.1p00*c2.0p00': ('GluGluToHH_kl5p00_kt1p00_c20p00', 0.09965e3*0.00227*0.582*2), 
    'Run3*GluGluToHH*kl.5p00*kt.1p00*c2.0p00': ('GluGluToHH_kl5p00_kt1p00_c20p00', 0.09965e3*0.00227*0.582*2), 

    'Run2*GluGluToHH*kl.0p00*kt.1p00*c2.0p00': ('GluGluToHH_kl0p00_kt1p00_c20p00', 0.07575e3*0.00227*0.582*2), 
    'Run3*GluGluToHH*kl.0p00*kt.1p00*c2.0p00': ('GluGluToHH_kl0p00_kt1p00_c20p00', 0.07575e3*0.00227*0.582*2), 

    'Run2*GluGluToHH*kl.2p45*kt.1p00*c2.0p00': ('GluGluToHH_kl2p45_kt1p00_c20p00', 0.01477e3*0.00227*0.582*2), 
    'Run3*GluGluToHH*kl.2p45*kt.1p00*c2.0p00': ('GluGluToHH_kl2p45_kt1p00_c20p00', 0.01477e3*0.00227*0.582*2), 

    'Run2*VBFHH*CV.1*C2V.1*C3.1': ('VBFToHH_cv1p00_c2v1p00_c31p00', 1.684*0.0026),
    'Run3*VBFHH*CV.1*C2V.1*C3.1': ('VBFToHH_cv1p00_c2v1p00_c31p00', 1.870*0.0026),

    ## Resonant (Mgg) background ##
    # Fake b-jets #
    'Run2*GluGluHToGG': ('GluGluHToGG', 0.1103e3),
    'Run3*GluGluHToGG': ('GluGluHToGG', 52170.*0.00228), 

    'Run2*VBFHToGG': ('VBFHToGG', 0.00855e3),
    'Run3*VBFHToGG': ('VBFHToGG', 4075.*0.00228),

    'Run3*Wm*HToGG': ('VHToGG', 566.4*0.00228), 

    'Run3*Wp*HToGG': ('VHToGG', 887.*0.00228),

    # Real b-jets #
    'Run2*ttHToGG': ('ttHToGG', 0.0011e3),
    'Run3*ttHToGG': ('ttHToGG', 568.8*0.00228), 

    'Run3*bbHToGG': ('bbHToGG', 525.1*0.00228),

    # Resonant b-jets #
    'Run2*VHToGG': ('VHToGG', 0.00508e3),
    'Run3*VHToGG': ('VHToGG', (566.4 + 887. + 942.2)*0.00228), 

    'Run3*ZHToGG': ('VHToGG', 942.2*0.00228),

    ## Non-resonant (Mgg) background ##
    # Fake b-jets #
    'Run2*DDQCD*GGJets': ('GGJets', 86.96e3),
    'Run3*DDQCD*GGJets': ('GGJets', 88750.),

    # Real b-jets #
    'Run2*TTGG': ('TTGG', 0.01696e3),
    'Run3*TTGG': ('TTGG', 16.96),
    
    'Run3*SherpaNLO': ('GGBBJets', 1093.),

    # Data-driven background #
    'DDQCD*DDQCDGJet': ('DDQCDGJets', 1),
                        
    # Data #
    'Data': ('Data', 1)
}
# Sample reweighting
sample_era_reweighting = {
    "2016*preVFP*DDQCD*DDQCDGJet": 1.1233,
    "2016*preVFP*DDQCD*GGJets": 1.1303,
    
    "2016*postVFP*DDQCD*DDQCDGJet": 1.2077,
    "2016*postVFP*DDQCD*GGJets": 1.3069,
    
    "2017*DDQCD*DDQCDGJet": 1.1817, 
    "2017*DDQCD*GGJets": 1.2002,
    
    "2018*DDQCD*DDQCDGJet": 1.1495, 
    "2018*DDQCD*GGJets": 1.2153,
    
    "2022*preEE*DDQCD*DDQCDGJet": 1.0498,
    "2022*preEE*DDQCD*GGJets": 1.3980,
    
    "2022*postEE*DDQCD*DDQCDGJet": 1.0896,
    "2022*postEE*DDQCD*GGJets": 1.3762,
    
    "2023*preBPix*DDQCD*DDQCDGJet": 1.0369,
    "2023*preBPix*DDQCD*GGJets": 1.4987,
    
    "2023*postBPix*DDQCD*DDQCDGJet": 1.1814,
    "2023*postBPix*DDQCD*GGJets": 1.5032,
    
    "2024*DDQCD*DDQCDGJet": 1.2140,
    "2024*DDQCD*GGJet": 1.5925,

    "!DDQCDGJet*GJet": 2.6,

    # Matches to everything, so returns 1 for sample-eras that don't match to the other keys
    "": 1
}

################################


def add_basic_info(df: pd.DataFrame, filepath):
    if '/sim/' in filepath: datatype = 'MC'
    elif '/data/' in filepath: datatype = 'Data'  
    else: raise Exception(f"Expecting \'sim\' or \'data\' in filepath, unclear what type of sample is input")

    # Add useful parquet meta-info
    df['sample_name'] = match_sample_name(filepath, xs_name_map)
    df['sample_era'] = match_sample_era(filepath)
    print(f"{match_sample_era(filepath)}: {match_sample_name(filepath, xs_name_map)}")

    if datatype.upper() == 'MC':
        print(f"lumi match = {match_sample(filepath, luminosities.keys())}: {match_sample_lumi(filepath, luminosities)}")
        print(f"xs match = {match_sample(filepath, xs_name_map.keys())}: {match_sample_xs(filepath, xs_name_map):.4f}")

        df['eventWeight'] = df['weight'] * match_sample_lumi(filepath, luminosities) * match_sample_xs(filepath, xs_name_map)
        if match_sample_name(filepath, xs_name_map) == 'DDQCDGJets': df['eventWeight'] = df['weight']
        
        df['eventWeight'] = df['eventWeight'] * sample_era_reweighting[match_sample(filepath, sample_era_reweighting.keys())]
    else: 
        df['weight'] =  np.ones(len(df))
        df['eventWeight'] = np.ones(len(df))

    if 'hash' not in df.columns:
        df['hash'] = np.arange(len(df))
