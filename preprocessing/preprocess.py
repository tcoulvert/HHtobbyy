import argparse
import glob
import logging
import os, subprocess

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector as vec
vec.register_awkward()

from resolved_preprocessing import  add_vars_resolved
from boosted_preprocessing import add_vars_boosted
from preprocessing_utils import match_sample, get_era_filepaths


################################

# MC Era: total era luminosity [fb^-1] #
luminosities = {
    '2022*preEE': 7.9804,
    '2022*postEE': 26.6717,
    '2023*preBPix': 17.794,
    '2023*postBPix': 9.451,
    '2024': 109.08
}
# Name: cross section [fb] @ sqrrt{s}=13.6 TeV & m_H=125.09 GeV #
cross_sections = {
    # Signal #
    'GluGluToHH': 34.43*0.0026, 
    # 'VBFHH': 1.870*0.0026,

    # Resonant (Mgg) background #
    # Fake b-jets
    'GluGluHToGG': 52170*0.00228, 'VBFHToGG': 4075*0.00228, 'W*HToGG': 1453*0.00228,
    'W*HTo2G': 1453*0.00228*0.6741,
    # Real b-jets
    'ttHToGG': 568.8*0.00228, 'bbHToGG': 525.1*0.00228,
    # Resonant b-jets
    'VHToGG': (2*1453 + 942.2)*0.00228, 'ZHToGG': 942.2*0.00228,
    'ZH*To2G': 942.2*0.00228*0.69911,

    # Non-resonant (Mgg) background #
    # Fake photons, fake b-jets
    'GJet*20to40*': 242500., 'GJet*40*': 919100., 
    # Fake photons
    'TTG*10to100': 4216., 'TTG*100to200': 411.4, 'TTG*200': 128.4,
    # Fake b-jets
    'GGJets*40to80': 318100., 'GGJets*80': 88750.,
    # Real b-jets
    'TTGG': 16.96,
}
sample_name_map = {
    # Signal #
    'GluGluToHH', 'VBFToHH',

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

    # Data #
    'Data'
}
BAD_DIRS = {'outdated'}

RUN_ALL_MC = False
END_FILEPATH = "*merged.parquet"
NEW_END_FILEPATH = "*preprocessed.parquet"

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

################################


def get_files(eras, type='MC'):
    for era in eras.keys():
        all_dirs_set = set(
            glob.glob(os.path.join(era, "**", END_FILEPATH), recursive=True)
        )

        # Remove bad dirs
        all_dirs_set = set(
            item for item in all_dirs_set 
            if match_sample(item, BAD_DIRS) is None
        )

        # Remove non-necessary MC samples
        if type.upper() == 'MC' and not RUN_ALL_MC:
            all_dirs_set = set(
                item for item in all_dirs_set 
                if match_sample(item, cross_sections.keys()) is not None
            )
        

        eras[era] = sorted(all_dirs_set)

def get_output_filepath(input_filepath: str):
    keep_substr = input_filepath[:input_filepath.rfind(END_FILEPATH[1:])]
    new_substr = NEW_END_FILEPATH[1:]
    return keep_substr+new_substr

def make_dataset(filepath, era, type='MC'):
    print('======================== \n', 'Starting \n', filepath)
    pq_file = pq.ParquetFile(filepath)
    schema = pq.read_schema(filepath)
    columns = [
        field for field in schema.names if (
            'VBF' not in field
            and not (
                'nonResReg' in field and 'nonResReg_DNNpair' not in field
            )
        )
    ]

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
            ak_batch['eventWeight'] = (
                ak_batch['weight'] 
                * luminosities[match_sample(filepath, luminosities.keys())] 
                * cross_sections[match_sample(filepath, cross_sections.keys())]
            )
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
    # subprocess.run(["rm", "-f", filepath])
    # subprocess.run(["mv", output_filepath, filepath])
    print('Finished \n', '========================')

def make_mc(sim_eras=None, output=None):
    if sim_eras is None:
        logger.exception(
            "Not processing any MC files, returning with status code 1"
        )
        return 1
    if output is not None and not os.path.exists(output):
        os.makedirs(output)
    
    # Pull MC sample dir_list
    get_files(sim_eras)
    
    # Perform the variable calculation and merging
    for sim_era, filepaths in sim_eras.items():
        for filepath in filepaths:
            if match_sample(filepath, {'_up/', '_down/'}) is not None: continue
            make_dataset(filepath, sim_era)

def make_data(data_eras=None, output=None):
    if data_eras is None:
        logger.exception(
            "Not processing any Data files, returning with status code 1"
        )
        return 1
    
    # Pull Data sample dir_list
    get_files(data_eras, type='Data')

    # Perform the variable calculation and merging
    for data_era, filepaths in data_eras.items():
        for filepath in filepaths:
            make_dataset(filepath, data_era, type='Data')

if __name__ == '__main__':
    args = parser.parse_args()

    SIM_ERAS, DATA_ERAS = get_era_filepaths(args.input_eras, split_data_mc_eras=True)

    sim_eras = {
            os.path.join(era, ''): list() for era in SIM_ERAS
    } if args.sim_era_filepaths is not None else None
    make_mc(sim_eras=sim_eras, output=args.output_dirpath)

    data_eras = {
            os.path.join(era, ''): list() for era in DATA_ERAS
    } if args.data_era_filepaths is not None else None
    make_data(data_eras=data_eras, output=args.output_dirpath)
