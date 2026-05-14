# Stdlib packages
import math
import os
import re

# HEP packages
import awkward as ak
import eos_utils as eos
import pyarrow.parquet as pq

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import FILL_VALUE, match_sample, get_era_filepaths

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
    try: pq.read_schema(parquet_filepath); return True
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
    if not os.path.exists('/'.join(output_filepath.split('/')[:-1])):
        os.makedirs('/'.join(output_filepath.split('/')[:-1]))
    return output_filepath


def match_sample_name(filepath: str, sample_name_map: dict):
    for filepath_piece in reversed(filepath.split('/')):
        piece_match = match_sample(filepath_piece, sample_name_map.keys())
        if piece_match is not None: return sample_name_map[piece_match]
    return filepath