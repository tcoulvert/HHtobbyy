# Stdlib packages
import argparse
import copy
import json
import logging
import os
import subprocess
import sys

# Common Py packages
import numpy as np
import pandas as pd
import prettytable as pt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# HEP packages
import hist

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.Categorization import CategorizationConfig
# from HHtobbyy.event_discrimination.Model import Model
# from HHtobbyy.workspace_utils.retrieval_utils import match_regex, match_sample
# from HHtobbyy.event_discrimination.evaluation import transform_preds_options, transform_preds
# from HHtobbyy.categorization.categorization_utils import *

################################

class Categorization:
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.catconfig = CategorizationConfig(dfdataset, config)
