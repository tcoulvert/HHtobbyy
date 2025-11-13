import numpy as np
import pandas as pd
import pickle
import mplhep as hep
import matplotlib.pyplot as plt
import yaml
import math
from tqdm import tqdm
plt.style.use(hep.style.CMS)
import uproot3 as uproot
import itertools
import os
import importlib.util
import sys
from array import array
import matplotlib.colors as mcolors
import gc
import glob

preEE=7.98/61.9
postEE=26.67/61.9
preBPix=17.79/61.9
postBPix=9.45/61.9
_systematics=["JEC"] #"JER","ScaleEB","ScaleEE","Smearing"]
_dir=["Down"]
weight_cols = ['weight']

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