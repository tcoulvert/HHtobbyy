# Stdlib packages
import math
import os
import re

# HEP packages
import awkward as ak
import eos_utils as eos
import pyarrow.parquet as pq

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import FILL_VALUE, match_sample

################################


################################
# Math functions
def ak_sign(ak_array, inverse=False):
    if not inverse:
        return ak.where(ak_array < 0, -1, 1)
    else:
        return ak.where(ak_array < 0, 1, -1)
        
def ak_abs(ak_array):
    valid_entry_mask = ak.where(ak_array != FILL_VALUE, True, False)
    abs_ak_array = ak.where(ak_array > 0, ak_array, -ak_array)
    return ak.where(valid_entry_mask, abs_ak_array, FILL_VALUE)
    
def deltaPhi(phi1, phi2):
    # angle1 and angle2 are (-pi, pi]
    # Convention: clockwise is (+), anti-clockwise is (-)
    subtract_angles = phi1 - phi2
    return ak.where(ak_abs(subtract_angles) <= math.pi, subtract_angles, subtract_angles + 2*math.pi*ak_sign(subtract_angles, inverse=True))

def deltaEta(eta1, eta2):
    return ak_abs(eta1 - eta2)


################################
# File functions
def has_magic_bytes(parquet_filepath: str):
    try: 
        eos_filepath = eos.load_file_eos(parquet_filepath)
        pq.read_schema(eos_filepath)
        eos.delete_lockfile(eos_filepath); return True
    except: return False

def get_output_filepath(input_filepath: str, output_dirpath: str|None, end_filepaths: str|None, new_end_filepath: str|None, base_filepath: str|None):
    if output_dirpath is None:
        output_filepath = input_filepath.replace(match_sample(input_filepath, end_filepaths), new_end_filepath)
    else:
        output_filepath = os.path.join(
            output_dirpath, 
            input_filepath[
                re.search(base_filepath, input_filepath).start():
            ].replace(match_sample(input_filepath, end_filepaths), new_end_filepath)
        )
    return output_filepath


def match_sample_name(filepath: str, xs_sample_map: dict):
    match = match_sample(filepath, xs_sample_map.keys())
    if match is not None: return xs_sample_map[match][0]
    else: return filepath

def match_sample_xs(filepath: str, xs_sample_map: dict):
    match = match_sample(filepath, xs_sample_map.keys())
    if match is not None: return xs_sample_map[match][1]
    else: return None

def match_sample_lumi(filepath: str, luminosities: dict):
    match = match_sample(filepath, luminosities.keys())
    if match is not None: return luminosities[match]
    else: return None

def match_sample_era(era: str, era_basestr: str='Run[1-3]_20'):
    match = re.search(era_basestr, era)
    if match is not None: return era[match.start():-1]
    else: return era
