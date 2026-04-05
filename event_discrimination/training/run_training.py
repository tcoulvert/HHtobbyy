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
import xgboost as xgb

# Workspace packages
from HHtobbyy.event_discrimination.models import Model
from HHtobbyy.event_discrimination.training.condor_training import submit

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Run BDT training")
parser.add_argument(
    "model", type=Model, help="Model to perform the training on"
)
parser.add_argument(
    "--optimize_space",  
    action="store_true",
    help="Boolean to do hyperparameter optimization (VERY EXPENSIVE)"
)
parser.add_argument(
    "--batch", 
    choices=["iterative", "condor"], 
    default="iterative",
    help="Flag to submit training as batch jobs"
)
parser.add_argument(
    "--memory", 
    type=str,
    default="10GB",
    help="Memory to request on batch jobs"
)
parser.add_argument(
    "--queue", 
    choices=["workday", "longlunch"], 
    default="workday",
    help="Queue to request on batch jobs"
)


################################

def run_training(model: Model, condor_config: dict={}):
    for fold in range(model.dfdataset.n_folds): model.train(fold)

