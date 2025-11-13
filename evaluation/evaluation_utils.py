# %matplotlib widget
# Stdlib packages
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# HEP packages
import gpustat
import pyarrow.parquet as pq
import xgboost as xgb

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing/"))


################################


def get_ttH_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])

def get_QCD_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])
