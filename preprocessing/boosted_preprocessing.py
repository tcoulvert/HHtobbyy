import copy
import glob
import json
import math
import os
import re

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector as vec
vec.register_awkward()

from preprocessing_utils import (
    ak_sign, ak_abs, deltaPhi, deltaEta, 
    match_sample
)

################################


FILL_VALUE = -999
NUM_FATJETS = 6

boosted_bbTagWPs = {
    # '2022*preEE': {'L': 0.047, 'M': 0.245, 'T': 0.6734, 'XT': 0.7862, 'XXT': 0.961},
    # '2022*postEE': {'L': 0.0499, 'M': 0.2605, 'T': 0.6915, 'XT': 0.8033, 'XXT': 0.9664},
    # '2023*preBPix': {'L': 0.0358, 'M': 0.1917, 'T': 0.6172, 'XT': 0.7515, 'XXT': 0.9659},
    # '2023*postBPix': {'L': 0.0359, 'M': 0.1919, 'T': 0.6133, 'XT': 0.7544, 'XXT': 0.9688},
    # '2024': {'L': 0.0246, 'M': 0.1272, 'T': 0.4648, 'XT': 0.6298, 'XXT': 0.9739}
}

################################


# Variables to add for boosted training
def add_bTagWP_boosted(sample, era):
    pass
    # WP_dict = boosted_bbTagWPs[match_sample(era, boosted_bbTagWPs.keys())]
    
    # for AN_type in ['nonRes', 'nonResReg', 'nonResReg_DNNpair']:
    #     for bjet_type in ['lead', 'sublead']:
    #         for WPname, WP in WP_dict.items():
    #             sample[f"{AN_type}_{bjet_type}_bjet_bTagWP{WPname}"] = ak.where(
    #                 sample[f"{AN_type}_{bjet_type}_bjet_btagPNetB"] > WP,
    #                 1, 0
    #             )

def add_vars_boosted(sample, era):
    pass
    # add_bTagWP_boosted(sample, era)

