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
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing", ""))
sys.path.append(os.path.join(GIT_REPO, "models", ""))


import training_utils as utils
import optimize_hyperparams_DNN as optDNN
from retrieval_utils import (
    get_train_Dataframes, get_labelND,
    get_class_sample_map, get_n_folds
)
from condor_training import submit
from mlp import MLP
from mlp_dataset import MLP_Dataset

################################


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Run BDT training")
parser.add_argument(
    "dataset_dirpath", 
    type=str,
    help="Full filepath to standardized BDT dataset"
)
parser.add_argument(
    "--output_dirpath", 
    type=str,
    default=os.path.join(GIT_REPO, "MultiClassDNN_model_outputs", ""),
    help="Full filepath to BDT output directory"
)
parser.add_argument(
    "--eos_dirpath",  
    default=os.path.join('root://cmseos.fnal.gov//', 'store', 'user', subprocess.run(['whoami'], capture_output=True, text=True).stdout.strip(), 'condor_train'),
    help="Dirpath for EOS space to store intermediate files onto"
)
parser.add_argument(
    "--optimize_space",  
    action="store_true",
    help="Boolean to do hyperparameter optimization (VERY EXPENSIVE)"
)
parser.add_argument(
    "--fold", 
    type=int,
    default=None,
    help="Only run training for a specific fold"
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


args = parser.parse_args()

CWD = str(Path().absolute())
DATASET_DIRPATH = os.path.join(args.dataset_dirpath, "")
BASE_DIRPATH = os.path.join(args.output_dirpath, "")
EOS_DIRPATH = os.path.join(args.eos_dirpath, "")
OPTIMIZE_SPACE = args.optimize_space
FOLD = args.fold
BATCH = args.batch
MEMORY = args.memory
QUEUE = args.queue

CLASS_SAMPLE_MAP = get_class_sample_map(DATASET_DIRPATH)
N_CLASSES = len(CLASS_SAMPLE_MAP)
N_FOLDS = get_n_folds(DATASET_DIRPATH)

VERSION = DATASET_DIRPATH.split('/')[-3]
VARS = '_'.join(DATASET_DIRPATH.split('/')[-2].split('_')[:-2])
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUTPUT_DIRPATH = os.path.join(BASE_DIRPATH, VERSION, VARS, CURRENT_TIME)
if not os.path.exists(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

################################


def run_training():
    # txt file pointing to location of standardized dataset used for training
    #  and therefore the default location for testing
    dataset_filepath = os.path.join(OUTPUT_DIRPATH, "dataset_filepath.txt")
    with open (dataset_filepath, "w") as f:
        f.write(DATASET_DIRPATH)
    
    # copy the standardization json file to output dir
    subprocess.run(
        ['cp', os.path.join(DATASET_DIRPATH, "standardization.json"), OUTPUT_DIRPATH],
        check=True
    )

    # Dict of hyperparameters for the model -- necessary to store for evaluation
    param_filepath = os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_best_params.json')
    if OPTIMIZE_SPACE:  # Need to change this to use condor as well..
        print('OPTIMIZING SPACE')
        
        param = optDNN.optimize_hyperparams(
            DATASET_DIRPATH, param_filepath, verbose=True
        )
    else:
        param = optDNN.init_params()

    # Train the model
    if BATCH == "iterative":
        for fold_idx in range(N_FOLDS):
            if FOLD is not None and FOLD != fold_idx: continue
            print(f"fold {fold_idx}")

            # Dataset
            train_df, train_aux, val_df,  val_aux, _, _ = get_train_Dataframes(DATASET_DIRPATH, fold_idx)
            train_data = DataLoader(
                MLP_Dataset(train_df.to_numpy(), get_labelND(train_aux['AUX_label1D'].to_numpy()), train_aux['AUX_eventWeightTrain'].to_numpy()),
                batch_size=param['batch_size'], shuffle=True, num_workers=param['num_workers']
            )
            val_data = DataLoader(
                MLP_Dataset(val_df.to_numpy(), get_labelND(val_aux['AUX_label1D'].to_numpy()), val_aux['AUX_eventWeightTrain'].to_numpy()),
                batch_size=param['batch_size'], shuffle=True, num_workers=param['num_workers']
            )

            # DNN model
            model = MLP(train_df.shape[1], param['num_layers'], param['num_nodes'], N_CLASSES, param['dropout_prob'])

            # Callbacks
            callbacks = [EarlyStopping(monitor=param['monitor'], min_delta=param['min_delta'], patience=param['patience'], verbose=False, mode=param['mode'])]

            # Build trainer
            trainer = Trainer(
                callbacks=callbacks,
                default_root_dir=OUTPUT_DIRPATH,
                max_epochs=param['max_epochs'], 
                accelerator=param['accelerator'],
                strategy=param['strategy'],
                num_nodes=param['num_nodes'],
                precision=param['precision'], 
                gradient_clip_val=param['gradient_clip_val'],
                logger=param['logger']
            )

            # Train DNN
            trainer.fit(model, train_data, val_data)

    elif BATCH == "condor":
        pass

if __name__ == "__main__":
    run_training()
