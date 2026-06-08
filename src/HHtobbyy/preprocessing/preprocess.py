# Stdlib packages
import argparse
import glob
import logging
import os
import re
import subprocess

# Common Py packages
import numpy as np

# HEP packages
import awkward as ak
import eos_utils as eos
import pyarrow
import pyarrow.parquet as pq
import vector as vec

# Workspace packages
from HHtobbyy.preprocessing.resolved_preprocessing import add_vars_resolved
from HHtobbyy.preprocessing.boosted_preprocessing import add_vars_boosted
from HHtobbyy.preprocessing.preprocessing_utils import (
    match_sample_name, match_sample_xs, match_sample_lumi, match_sample_era, get_output_filepath, has_magic_bytes
)
from HHtobbyy.workspace_utils.retrieval_utils import match_sample, get_era_filepaths

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Preprocess data to add necessary BDT variables.")
parser.add_argument(
    "input_eras",
    help="File for input eras to run processing"
)
parser.add_argument(
    "--output_dirpath", 
    default=None,
    help="Full filepath on LPC for output to be dumped, default is for each new file to be adjacent to input files, but with slightly changed filenames."
)
parser.add_argument(
    "--base_filepath", 
    default='Run._20..',
    help="Regex string for splitting filepath if using a new output_dirpath"
)
parser.add_argument(
    "--batch_size", 
    type=int,
    default=1_048_576,
    help="Batch size for batch reading/writing of parquets"
)
parser.add_argument(
    "--force", 
    action="store_true",
    help="Boolean flag to rerun processing on files that already exist, defaults to only running on files that haven't been run"
)
parser.add_argument(
    "--run_all_mc", 
    action="store_true",
    help="Boolean flag to run processing on all MC files, defaults to only running files matching keys in `cross_sections` dict"
)

################################


vec.register_awkward()

args = parser.parse_args()
INPUT_ERAS = args.input_eras
OUTPUT_DIRPATH = args.output_dirpath
BASE_FILEPATH = args.base_filepath
BATCH_SIZE = args.batch_size
FORCE = args.force
RUN_ALL_MC = args.run_all_mc

BAD_DIRS = {'outdated', 'allData'}
END_FILEPATHS = ["merged.parquet", "Rescaled.parquet"]
NEW_END_FILEPATH = "preprocessed.parquet"

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


def get_files(eras, type='MC'):
    for era in eras.keys():
        glob_dirs_set = lambda end_filepath: set(
            eos.glob_eos(os.path.join(era, "**", f"*{end_filepath}"), recursive=True)
        )
        all_dirs_set = set(elem for end_filepath in END_FILEPATHS for elem in glob_dirs_set(end_filepath))

        if OUTPUT_DIRPATH is None:
            ran_dirs_set = set(
                parquet_filepath for parquet_filepath in glob_dirs_set(NEW_END_FILEPATH) 
                if has_magic_bytes(parquet_filepath)
            )
        else:
            ran_dirs_set = set(
                parquet_filepath 
                for parquet_filepath in eos.glob_eos(os.path.join(OUTPUT_DIRPATH, "**", f"*{NEW_END_FILEPATH}"), recursive=True)
                if has_magic_bytes(parquet_filepath)
            )

        # Remove bad dirs
        all_dirs_set = set(
            item for item in all_dirs_set 
            if match_sample(item, BAD_DIRS) is None
        )
        if not FORCE:
            all_dirs_set = set(
                input_filepath for input_filepath in all_dirs_set 
                if get_output_filepath(input_filepath, OUTPUT_DIRPATH, END_FILEPATHS, NEW_END_FILEPATH, BASE_FILEPATH) not in ran_dirs_set
            )

        # Remove non-necessary MC samples
        if type.upper() == 'MC' and not RUN_ALL_MC:
            all_dirs_set = set(
                item for item in all_dirs_set 
                if match_sample(item, xs_name_map.keys()) is not None
            )

        eras[era] = sorted(all_dirs_set)

def make_dataset(filepath, era, type='MC'):
    print('========================>\n'+'Starting \n', filepath)
    pq_file = pq.read_parquet(eos.load_file_eos(filepath))
    schema = pq.read_schema(filepath)
    columns = [field for field in schema.names]

    output_filepath = eos.save_file_eos(get_output_filepath(filepath, OUTPUT_DIRPATH, END_FILEPATHS, NEW_END_FILEPATH, BASE_FILEPATH))
    pq_writer = None
    for pq_batch in pq_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        ak_batch = ak.from_arrow(pq_batch)

        # Add useful parquet meta-info
        ak_batch['sample_name'] = match_sample_name(filepath, xs_name_map)
        ak_batch['sample_era'] = match_sample_era(era)
        print(f"{match_sample_era(era)}: {match_sample_name(filepath, xs_name_map)}")
        if type.upper() == 'MC':
            print(f"lumi match = {match_sample(filepath, luminosities.keys())}: {match_sample_lumi(filepath, luminosities)}")
            print(f"xs match = {match_sample(filepath, xs_name_map.keys())}: {match_sample_xs(filepath, xs_name_map):.4f}")
            ak_batch['eventWeight'] = ak_batch['weight'] * match_sample_lumi(filepath, luminosities) * match_sample_xs(filepath, xs_name_map)
            if match_sample_name(filepath, xs_name_map) == 'DDQCDGJets': ak_batch['eventWeight'] = ak_batch['weight']
            ak_batch['eventWeight'] = ak_batch['eventWeight'] * sample_era_reweighting[match_sample(filepath, sample_era_reweighting.keys())]
        else: 
            ak_batch['weight'] =  ak.ones_like(ak_batch['pt'])
            ak_batch['eventWeight'] =  ak.ones_like(ak_batch['pt'])

        print('Adding resolved vars')
        add_vars_resolved(ak_batch, filepath)
        print('Finished resolved vars, adding boosted vars')
        add_vars_boosted(ak_batch, filepath)
        print('Finished boosted vars')
        if 'hash' not in ak_batch.fields:
            ak_batch['hash'] = np.arange(ak.num(ak_batch['pt'], axis=0))

        table_batch = ak.to_arrow_table(ak_batch)
        if pq_writer is None: pq_writer = pq.ParquetWriter(output_filepath, schema=table_batch.schema)
        pq_writer.write_table(table_batch)
    if pq_writer is not None: pq_writer.close()
    print('Finished\n'+'<========================')

def make_mc(sim_eras: dict):
    if sim_eras is None:
        logger.log(1, "Not processing any MC files"); return
    
    # Pull MC sample dir_list
    get_files(sim_eras)
    
    # Perform the variable calculation and merging
    for sim_era, filepaths in sim_eras.items():
        for filepath in filepaths:
            if match_sample(filepath, {'_up/', '_down/'}) is not None: continue
            make_dataset(filepath, sim_era)

def make_data(data_eras: dict):
    if data_eras is None:
        logger.log(1, "Not processing any Data files"); return
    
    # Pull Data sample dir_list
    get_files(data_eras, type='Data')

    # Perform the variable calculation and merging
    for data_era, filepaths in data_eras.items():
        for filepath in filepaths:
            make_dataset(filepath, data_era, type='Data')

if __name__ == '__main__':
    SIM_ERAS, DATA_ERAS = get_era_filepaths(INPUT_ERAS, split_data_mc_eras=True)

    sim_eras = {
        os.path.join(era, ''): list() for era in SIM_ERAS
    } if len(SIM_ERAS) > 0 else None
    make_mc(sim_eras)

    data_eras = {
        os.path.join(era, ''): list() for era in DATA_ERAS
    } if len(DATA_ERAS) > 0 else None
    make_data(data_eras)

