# Stdlib packages
import argparse
import glob
import logging
import os
import re

# Common Py packages
import numpy as np

# HEP packages
import awkward as ak
import pyarrow.parquet as pq
import vector as vec

################################


from resolved_preprocessing import  add_vars_resolved
from boosted_preprocessing import add_vars_boosted
from preprocessing_utils import match_sample, get_era_filepaths

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
    default='Run3_202',
    help="Regex string for splitting filepath if using a new output_dirpath"
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
if OUTPUT_DIRPATH is not None and not os.path.exists(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)
BASE_FILEPATH = args.base_filepath
FORCE = args.force
RUN_ALL_MC = args.run_all_mc

BAD_DIRS = {'outdated'}
END_FILEPATHS = ["merged.parquet", "Rescaled.parquet"]
NEW_END_FILEPATH = "preprocessed.parquet"

# MC Era: total era luminosity [fb^-1] #
luminosities = {
    '2022*preEE': 7.9804,
    '2022*postEE': 26.6717,
    '2023*preBPix': 17.794,
    '2023*postBPix': 9.451,
    '2024': 109.08,
    '*****DDQCDGJets*****': 1.
}
# Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
cross_sections = {
    # Signal #
    'GluGluToHH': 34.43*0.0026, 
    'VBFHH': 1.870*0.0026,

    # Resonant (Mgg) background #
    # Fake b-jets
    'GluGluHToGG': 52170.*0.00228, 'VBFHToGG': 4075.*0.00228, 
    'Wm*HToGG': 566.4*0.00228, 'Wp*HToGG': 887.*0.00228,
    'Wm*HTo2G': 566.4*0.00228*0.6741, 'Wp*HTo2G': 887.*0.00228*0.6741,  # Irene's samples
    # Real b-jets
    'ttHToGG': 568.8*0.00228, 'bbHToGG': 525.1*0.00228,
    # Resonant b-jets
    'VHToGG': (566.4 + 887. + 942.2)*0.00228, 'ZHToGG': 942.2*0.00228,
    'ZH*To2G': 942.2*0.00228*0.69911,  # Irene's samples

    # Non-resonant (Mgg) background #
    # Fake photons, fake b-jets
    'GJet*20to40*': 242500., 'GJet*40*': 919100., 
    # Fake photons
    'TTG*10to100': 4216., 'TTG*100to200': 411.4, 'TTG*200': 128.4,
    # Fake b-jets
    'GGJets*40to80': 318100., 'GGJets*80': 88750.,
    # Real b-jets
    'TTGG': 16.96,

    # Data-driven background #
    'DDQCDGJets': 1.,
}
sample_name_map = {
    # Signal #
    'GluGluToHH', 'VBFHH',

    # Resonant (Mgg) background #
    # Fake b-jets
    'GluGluHToGG', 'VBFHToGG', 'WmHToGG', 'WpHToGG',
    # Real b-jets
    'ttHToGG', 'bbHToGG',
    # Resonant b-jets
    'VHToGG', 'ZHToGG',

    # Non-resonant (Mgg) background #
    # Fake photons, fake b-jets
    'GJet',
    # Fake photons
    'TTG',
    # Fake b-jets
    'GGJets',
    # Real b-jets
    'TTGG',

    # Data-driven background #
    'DDQCDGJets',

    # Data #
    'Data'
}

################################


def has_magic_bytes(parquet_filepath: str):
    try:
        pq.read_schema(parquet_filepath)
        return True
    except:
        return False

def get_files(eras, type='MC'):
    for era in eras.keys():
        glob_dirs_set = lambda end_filepath: set(
            glob.glob(os.path.join(era, "**", f"*{end_filepath}"), recursive=True)
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
                for parquet_filepath in glob.glob(os.path.join(OUTPUT_DIRPATH, "**", f"*{NEW_END_FILEPATH}"), recursive=True)
                if has_magic_bytes(parquet_filepath)
            ) & set(get_output_filepath(input_filepath) for input_filepath in all_dirs_set)

        # Remove bad dirs
        all_dirs_set = set(
            item for item in all_dirs_set 
            if match_sample(item, BAD_DIRS) is None
        )
        if not FORCE:
            all_dirs_set = all_dirs_set - set(elem.replace(NEW_END_FILEPATH, end_filepath) for elem in ran_dirs_set for end_filepath in END_FILEPATHS)

        # Remove non-necessary MC samples
        if type.upper() == 'MC' and not RUN_ALL_MC:
            all_dirs_set = set(
                item for item in all_dirs_set 
                if match_sample(item, cross_sections.keys()) is not None
            )

        eras[era] = sorted(all_dirs_set)

def get_output_filepath(input_filepath: str):
    if OUTPUT_DIRPATH is None:
        output_filepath = input_filepath.replace(match_sample(input_filepath, END_FILEPATHS), NEW_END_FILEPATH)
    else:
        output_filepath = os.path.join(
            OUTPUT_DIRPATH, 
            input_filepath[
                re.search(BASE_FILEPATH, input_filepath).start():
            ].replace(match_sample(input_filepath, END_FILEPATHS), NEW_END_FILEPATH)
        )
    if not os.path.exists('/'.join(output_filepath.split('/')[:-1])):
        os.makedirs('/'.join(output_filepath.split('/')[:-1]))
    return output_filepath

def make_dataset(filepath, era, type='MC'):
    print('======================== \n', 'Starting \n', filepath)
    pq_file = pq.ParquetFile(filepath)
    schema = pq.read_schema(filepath)
    columns = [field for field in schema.names]

    output_filepath = get_output_filepath(filepath)
    pq_writer = None
    for pq_batch in pq_file.iter_batches(batch_size=524_288, columns=columns):
        ak_batch = ak.from_arrow(pq_batch)

        # Add useful parquet meta-info
        ak_batch['sample_name'] = match_sample(filepath, sample_name_map) if match_sample(filepath, sample_name_map) is not None else filepath.split('/')[-3]
        ak_batch['sample_era'] = era[era.find('Run3_202'):-1]
        if type.upper() == 'MC':
            print(f"lumi match = {match_sample(filepath, luminosities.keys())}")
            print(f"xs match = {match_sample(filepath, cross_sections.keys())}")
            if match_sample(filepath, cross_sections.keys()) != 'DDQCDGJets':
                ak_batch['eventWeight'] = (
                    ak_batch['weight'] 
                    * luminosities[match_sample(filepath, luminosities.keys())] 
                    * cross_sections[match_sample(filepath, cross_sections.keys())]
                )
            else: 
                ak_batch['eventWeight'] = ak_batch['weight']
        else: 
            ak_batch['weight'] =  ak.ones_like(ak_batch['pt'])
            ak_batch['eventWeight'] =  ak.ones_like(ak_batch['pt'])

        add_vars_resolved(ak_batch, filepath)
        add_vars_boosted(ak_batch, filepath)
        if 'hash' not in ak_batch.fields:
            ak_batch['hash'] = np.arange(ak.num(ak_batch['pt'], axis=0))

        table_batch = ak.to_arrow_table(ak_batch)
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(output_filepath, schema=table_batch.schema)
        pq_writer.write_table(table_batch)
    print('Finished \n', '========================')

def make_mc(sim_eras: dict):
    if sim_eras is None:
        logger.log(1, "Not processing any MC files")
    
    # Pull MC sample dir_list
    get_files(sim_eras)
    
    # Perform the variable calculation and merging
    for sim_era, filepaths in sim_eras.items():
        for filepath in filepaths:
            if match_sample(filepath, {'_up/', '_down/'}) is not None: continue
            make_dataset(filepath, sim_era)

def make_data(data_eras: dict):
    if data_eras is None:
        logger.log(1, "Not processing any Data files")
    
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

