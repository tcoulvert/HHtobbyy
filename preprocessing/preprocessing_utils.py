# Stdlib packages
import math
import re

# Common Py packages
import numpy as np

# HEP packages
import awkward as ak
import vector as vec

################################


vec.register_awkward()

FILL_VALUE = -999

################################


def get_era_filepaths(input_eras: str, split_data_mc_eras: bool=False):
    MC_eras, Data_eras = set(), set()
    with open(input_eras, 'r') as f:
        for line in f:
            stdline = line.strip()
            if len(stdline) == 0 or stdline[0] == '#': continue

            if 'sim' in stdline.lower(): MC_eras.add(stdline)
            elif 'data' in stdline.lower(): Data_eras.add(stdline)
            else: raise KeyError(f"Era {stdline} does not seem to be MC or Data, check the filepath is correct")
    if split_data_mc_eras: return MC_eras, Data_eras
    else: return (MC_eras | Data_eras)

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

def match_sample(sample_str, regexes):
    for regex in sorted(regexes, key=len, reverse=True):
        regex_bools = []
        match_str = sample_str
        for exp in regex.split('*'):
            if len(exp) == 0: continue
            if (exp[0] != '!' and re.search(exp.lower(), match_str.lower()) is not None):
                regex_bools.append(True)
                match_str = match_str[re.search(exp.lower(), match_str.lower()).end():]
            elif (exp[0] == '!' and re.search(exp[1:].lower(), match_str.lower()) is None):
                regex_bools.append(True)
            else:
                regex_bools.append(False)
        if all(regex_bools):
            return regex
     