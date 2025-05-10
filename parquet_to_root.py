import copy
import glob
import os
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot3 as uproot


output_fileprefix = os.path.join(os.getcwd(), '../CombineFits/BoostedRootFiles')
if not os.path.exists(output_fileprefix):
    os.makedirs(output_fileprefix)
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v2/"
# lpc_filegroup = lambda s: f'Run3_{s}_mergedResolved_MultiBDT_output'
lpc_filegroup = lambda s: f'Run3_{s}_mergedBoosted_Cut_output'
LPC_FILEPREFIX_22 = os.path.join(lpc_fileprefix, lpc_filegroup('2022'), 'sim', '')
LPC_FILEPREFIX_23 = os.path.join(lpc_fileprefix, lpc_filegroup('2023'), 'sim', '')
LPC_FILEPREFIX_24 = os.path.join(lpc_fileprefix, lpc_filegroup('2024'), 'sim', '')
END_FILEPATH = '*output.parquet' if re.search('MultiBDT_output', LPC_FILEPREFIX_22) is not None else '*merged.parquet'

CORR_WEIGHT_SYSTS = [  # Up and Down
    'bTagSF_sys_lf', 'bTagSF_sys_hf', 
]
UNCORR_WEIGHT_SYSTS = [  # Up and Down
    'ElectronVetoSF', 'PreselSF', 'TriggerSF', 'Pileup',
    'bTagSF_sys_lfstats1', 'bTagSF_sys_lfstats2',
    'bTagSF_sys_hfstats1', 'bTagSF_sys_hfstats2',
]
# SYST_MAP = { # _up and _down
#     'Et_dependent_ScaleEB': 'EBScale', 'Et_dependent_ScaleEE': 'EEScale', 
#     'Et_dependent_Smearing': 'Smearing', 
#     'jec_syst_Total': 'JEC', 'jer_syst': 'JER'
# }
SYST_MAP = {}

SIGNAL_SAMPLES = [
    'GluGluToHH', 'VBFHH',
]
SINGLEH_SAMPLES = [ 
    'GluGluHToGG', 'ttHToGG', 'VBFHToGG', 'VHToGG', 'bbHToGG',
]

LUMINOSITIES = {
    "2022": 7.9804 + 26.6717,  # preEE + postEE
    "2023": 17.794 +  9.451,  # preBPix + postBPix
    "2024": 109.08,
}
LUMINOSITIES['total_lumi'] = sum(LUMINOSITIES.values())
LUMINOSITIES['total_MClumi'] = LUMINOSITIES['2022'] + LUMINOSITIES['2023']  # b/c no 2024 MC yet, so don't want to normalize by full lumi

MC_YEARS = ['2022', '2023']
DATA_YEARS = ['2022', '2023', '2024']

NOMINAL_VARIABLES = ['CMS_hgg_mass', 'dZ', 'weight', 'eventWeight']
for direction in ['Up', 'Down']:
    NOMINAL_VARIABLES.extend(['weight_'+corr_weight+direction for corr_weight in CORR_WEIGHT_SYSTS])
    for year in MC_YEARS:
        NOMINAL_VARIABLES.extend(['weight_'+year+'_'+uncorr_weight+direction for uncorr_weight in UNCORR_WEIGHT_SYSTS])
SYST_VARIABLES = ['CMS_hgg_mass', 'dZ', 'weight', 'eventWeight']
DATA_VARIABLES = ['CMS_hgg_mass']


def main():

    # MC dataframes
    MC_TTree_name = '_125_13p6TeV_cat0'
    MCDFs_dict = {}
    for year in MC_YEARS:
        file_prefix = os.path.join(lpc_fileprefix, lpc_filegroup(year), 'sim', '')

        for variation in ['nominal']+list(SYST_MAP.keys()):
            directions = [''] if variation == 'nominal' else ['_up', '_down']

            for direction in directions:
                syst_name = '' if variation == 'nominal' else '_'+SYST_MAP[variation]+direction[1:].upper()
                year_filepaths = glob.glob(os.path.join(file_prefix, "**", variation+direction, END_FILEPATH), recursive=True)
                
                for sample_name in SIGNAL_SAMPLES+SINGLEH_SAMPLES:
                    sample_filepaths = [year_filepath for year_filepath in year_filepaths if re.search(sample_name.lower(), year_filepath.lower()) is not None]
                    sample_filepaths.sort()

                    if len(sample_filepaths) < 1: continue

                    year_df = pd.concat([pd.json_normalize(pd.read_parquet(sample_filepath)['']) for sample_filepath in sample_filepaths], ignore_index=True)
                    year_df['CMS_hgg_mass'] = year_df['mass']  # Add Hgg mass variable

                    if variation == 'nominal':
                        for uncorr_weight in UNCORR_WEIGHT_SYSTS:

                            for direction in ['Up', 'Down']:
                                weight_syst_name = uncorr_weight+direction

                                for _year_ in MC_YEARS:
                                    if _year_ != year:
                                        year_df[f"weight_{_year_}_{weight_syst_name}"] = year_df[f"weight"]
                                    else:
                                        year_df[f"weight_{_year_}_{weight_syst_name}"] = year_df[f"weight_{weight_syst_name}"]

                        year_df = year_df[NOMINAL_VARIABLES]
                    else:
                        year_df = year_df[SYST_VARIABLES]

                    if f"{sample_name}{MC_TTree_name}{syst_name}" not in MCDFs_dict.keys():
                        MCDFs_dict[f"{sample_name}{MC_TTree_name}{syst_name}"] = copy.deepcopy(year_df)
                    else:
                        MCDFs_dict[f"{sample_name}{MC_TTree_name}{syst_name}"] = pd.concat([MCDFs_dict[f"{sample_name}{MC_TTree_name}{syst_name}"], year_df])

    for variation in ['nominal']+list(SYST_MAP.keys()):
        directions = [''] if variation == 'nominal' else ['_up', '_down']

        for direction in directions:
            syst_name = '' if variation == 'nominal' else '_'+SYST_MAP[variation]+direction[1:].upper()
                
            for sample_name in SINGLEH_SAMPLES:

                if f"singleH{MC_TTree_name}{syst_name}" not in MCDFs_dict.keys():
                    MCDFs_dict[f"singleH{MC_TTree_name}{syst_name}"] = copy.deepcopy(
                        MCDFs_dict[f"{sample_name}{MC_TTree_name}{syst_name}"]
                    )
                else:
                    MCDFs_dict[f"singleH{MC_TTree_name}{syst_name}"] = pd.concat([
                        MCDFs_dict[f"singleH{MC_TTree_name}{syst_name}"], MCDFs_dict[f"{sample_name}{MC_TTree_name}{syst_name}"]
                    ])

    for process in SIGNAL_SAMPLES+SINGLEH_SAMPLES+['singleH']:
        print(f'writing 2223_Boosted_{process}.root')
        with uproot.recreate(os.path.join(output_fileprefix, f"2223_Boosted_{process}.root")) as f:
            for key, df in MCDFs_dict.items():
                if re.match(process, key) is not None:
                    f[key] = uproot.newtree({col:'float64' for col in df.columns})
                    f[key].extend({col: df[col].to_numpy() for col in df.columns})
                    # if not np.all([re.search(syst, key) for syst in SYST_MAP.values()]):
                    #     Mgg_boundaries = (122.5, 127.5)
                    #     df_mask = np.logical_and(
                    #         df['CMS_hgg_mass'] > Mgg_boundaries[0],
                    #         df['CMS_hgg_mass'] < Mgg_boundaries[1]
                    #     )
                    #     yield_sum = np.sum(df.loc[df_mask, 'eventWeight'])
                    #     print(f"{process} yield within {Mgg_boundaries[0]:.1f} < Mgg < {Mgg_boundaries[1]:.1f} window: {yield_sum:.3f}")


    # # Data dataframes
    # Data_TTree_name = 'Data_13p6TeV_cat0'
    # DataDFs_dict = {}
    # for year in DATA_YEARS:
    #     file_prefix = os.path.join(lpc_fileprefix, lpc_filegroup(year), 'data', '')

    #     sample_filepaths = glob.glob(os.path.join(file_prefix, "**", END_FILEPATH), recursive=True)
    #     sample_filepaths.sort()

    #     if len(sample_filepaths) < 1: continue

    #     year_df = pd.concat([pd.json_normalize(pd.read_parquet(sample_filepath)['']) for sample_filepath in sample_filepaths], ignore_index=True)
    #     year_df['CMS_hgg_mass'] = year_df['mass']  # Add Hgg mass variable

    #     year_df = year_df[DATA_VARIABLES]

    #     if Data_TTree_name not in DataDFs_dict.keys():
    #         DataDFs_dict[Data_TTree_name] = copy.deepcopy(year_df)
    #     else:
    #         DataDFs_dict[Data_TTree_name] = pd.concat([DataDFs_dict[Data_TTree_name], year_df])

    # print(f'writing 2223_Boosted_Data.root')
    # with uproot.recreate(os.path.join(output_fileprefix, '2223_Boosted_Data.root')) as f:
    #     for key, df in DataDFs_dict.items():
    #         f[key] = uproot.newtree({col:'float64' for col in df.columns})
    #         f[key].extend({col: df[col].to_numpy() for col in df.columns})


if __name__ == '__main__':
    main()