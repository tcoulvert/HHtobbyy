import os
import glob
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot3 as uproot

preEE=7.98/61.9
postEE=26.67/61.9
preBPix=17.79/61.9
postBPix=9.45/61.9
_systematics=["JEC"] #"JER","ScaleEB","ScaleEE","Smearing"]
_dir=["Down"]
weight_cols = ['weight']

lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/"
# lpc_filegroup = lambda s: f'Run3_{s}_mergedResolved_MultiBDT_output'
lpc_filegroup = lambda s: f'Run3_{s}_mergedBoosted_Cut_output'
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, lpc_filegroup('2022'), 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, lpc_filegroup('2023'), 'sim', '')
LPC_FILEPREFIX_24 = os.path.join(lpc_fileprefix, lpc_filegroup('2024'), 'sim', '')
END_FILEPATH = '*output.parquet' if re.search('MultiBDT_output', LPC_FILEPREFIX_22) is not None else '*merged.parquet'

WEIGHT_SYSTS = [  # Up and Down
    'ElectronVetoSF', 'PreselSF', 'TriggerSF', 'Pileup',
    'bTagSF_sys_lf', 
    'bTagSF_sys_lfstats1', 'bTagSF_sys_lfstats2',
    'bTagSF_sys_hf', 
    'bTagSF_sys_hfstats1', 'bTagSF_sys_hfstats2',
]

VARIATION_SYSTS = [  # _up and _down
    'Et_dependent_ScaleEB', 'Et_dependent_ScaleEE', 
    'Et_dependent_Smearing', 
    'jec_syst_Total', 'jer_syst'
]

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


def main():
    # signal file
    signal_trees = {}
    for year in ['2022', '2023', '2024']:
        fileprefix = os.path.join(lpc_fileprefix, lpc_filegroup(year), 'sim', '')

        for variation in ['nominal']+VARIATION_SYSTS:
            directions = [''] if variation == 'nominal' else ['_up', '_down']
            for direction in directions:
                year_sample_filepaths = glob.glob(os.path.join(fileprefix, "**", variation+direction, END_FILEPATH), recursive=True)
                signal_sample_filepaths = [year_sample_filepath for year_sample_filepath in year_sample_filepaths if re.search('HH', year_sample_filepath)]
                signal_sample_filepaths.sort()




if __name__ == '__main__':
    main()


for str1 in _systematics:
    for str2 in _dir:
        source = str1
        direction = str2
        
        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/*VBFHH*.parquet")
        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE*1.76
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix*1.76
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix*1.76
        VBFHH = pd.concat(Temp, ignore_index=True)
        VBFHH['Signal'] = VBFHH["MultiBDT_output"].apply(lambda x: x[0] if len(x) > 1 else None)
        VBFHH['ttH'] = VBFHH["MultiBDT_output"].apply(lambda x: x[1] if len(x) > 1 else None)
        VBFHH['SingleH'] = VBFHH["MultiBDT_output"].apply(lambda x: x[2] if len(x) > 1 else None)
        VBFHH['NonResonant'] = VBFHH["MultiBDT_output"].apply(lambda x: x[3] if len(x) > 1 else None)
        VBFHH["2D_ttH_pred"]=VBFHH['Signal']/(VBFHH['Signal']+VBFHH['ttH'])
        VBFHH["2D_BDT_pred"]=VBFHH['Signal']/(VBFHH['Signal']+VBFHH['NonResonant']+VBFHH['SingleH'])      
        pklFile=open("2223Weighted/Variation/VBFHH_2223_%s%s.pkl"%(source,direction),"wb")
        pickle.dump(VBFHH,pklFile)
        pklFile.close()
        if 4>1:
            continue      
       # folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
       ##        "Variation/"+source+"/"+direction+"/2022preEE", 
       #        "Variation/"+source+"/"+direction+"/2023postBPix", 
       #        "Variation/"+source+"/"+direction+"/2023preBPix"]
       # parquet_files = []
       # for path in folder_paths:
       #     parquet_files += glob.glob(f"{path}/GluGlu*HH*.parquet")
       # Temp={}
       # for i in range(len(parquet_files)):
       #     tmp = pd.read_parquet(parquet_files[i])
       #     Temp[i] = pd.json_normalize(tmp[''])
       #     if i == 0:
       #         Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
       ##     if i == 1:
        #        Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
        #    if i == 2:
        #        Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
        #    if i == 3:
        #        Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        #Signal = pd.concat(Temp, ignore_index=True)
        #pklFile=open("2223Weighted/Variation/Signal_2223_%s%s.pkl"%(source,direction),"wb")
        #pickle.dump(Signal,pklFile)
        #pklFile.close()
        
        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/GluGluH*.parquet")
        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
            if i == 3:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        GGH = pd.concat(Temp, ignore_index=True)
        
        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/ttH*.parquet")

        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
            if i == 3:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        ttH = pd.concat(Temp, ignore_index=True)
        
        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/BBH*.parquet")
            parquet_files += glob.glob(f"{path}/bbH*.parquet")
        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
            if i == 3:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        BBH = pd.concat(Temp, ignore_index=True)
        
        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/VBFH*.parquet")
        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
            if i == 3:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        VBFH = pd.concat(Temp, ignore_index=True)

        folder_paths = ["Variation/"+source+"/"+direction+"/2022postEE", 
               "Variation/"+source+"/"+direction+"/2022preEE", 
               "Variation/"+source+"/"+direction+"/2023postBPix", 
               "Variation/"+source+"/"+direction+"/2023preBPix"]
        parquet_files = []
        for path in folder_paths:
            parquet_files += glob.glob(f"{path}/VH*.parquet")
        Temp={}
        for i in range(len(parquet_files)):
            tmp = pd.read_parquet(parquet_files[i])
            Temp[i] = pd.json_normalize(tmp[''])
            if i == 0:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postEE
            if i == 1:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preEE
            if i == 2:
                Temp[i][weight_cols]= Temp[i][weight_cols]*postBPix
            if i == 3:
                Temp[i][weight_cols]= Temp[i][weight_cols]*preBPix
        VH = pd.concat(Temp, ignore_index=True)
        
        GGH["Type"]=4.0
        VH["Type"]=5.0
        VBFH["Type"]=6.0
        ttH["Type"]=7.0
        BBH["Type"]=8.0
        MC2223=pd.concat([ttH, VH, GGH, VBFH, BBH],ignore_index=True)
        MC2223['Signal'] = MC2223["MultiBDT_output"].apply(lambda x: x[0] if len(x) > 1 else None)
        MC2223['ttH'] = MC2223["MultiBDT_output"].apply(lambda x: x[1] if len(x) > 1 else None)
        MC2223['SingleH'] = MC2223["MultiBDT_output"].apply(lambda x: x[2] if len(x) > 1 else None)
        MC2223['NonResonant'] = MC2223["MultiBDT_output"].apply(lambda x: x[3] if len(x) > 1 else None)
        MC2223["2D_ttH_pred"]=MC2223['Signal']/(MC2223['Signal']+MC2223['ttH'])
        MC2223["2D_BDT_pred"]=MC2223['Signal']/(MC2223['Signal']+MC2223['NonResonant']+MC2223['SingleH'])
        pklFile=open("2223Weighted/Variation/singleH_2223_%s%s.pkl"%(source,direction),"wb")
        pickle.dump(MC2223,pklFile)
        pklFile.close()