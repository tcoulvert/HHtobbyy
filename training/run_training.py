# Stdlib packages
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# HEP packages
import gpustat
import xgboost as xgb

################################

GIT_REPO = (
    subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .communicate()[0]
    .rstrip()
    .decode("utf-8")
)
sys.path.append(os.path.join(GIT_REPO, "preprocessing", ""))


import training_utils as utils
import optimize_hyperparams as opt
from retrieval_utils import (
    get_train_DMatrices, 
    get_class_sample_map, get_n_folds
)
from condor_training import submit

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
    default=os.path.join(GIT_REPO, "MultiClassBDT_model_outputs", ""),
    help="Full filepath to BDT output directory"
)
parser.add_argument(
    "--eos_dirpath",  
    default=os.path.join('root://cmseos.fnal.gov//', 'store', 'user', subprocess.run(['whoami'], capture_output=True, text=True).stdout, 'condor_train'),
    help="Dirpath for EOS space to store intermediate files onto"
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
    help="Memory to request on batch jobs"
)

################################


# LPC_FILEPREFIX = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/training_parquets/"
# PARQUET_TIME = "2025-11-18_19-49-19"  # 2022-23 WPs + high stats
# PARQUET_TIME = "2025-11-18_13-46-20"  # 2022-23 WPs + high stats + 3XT + 4XT
# PARQUET_TIME = "2025-11-18_14-54-32"  # 2022-24 WPs
# PARQUET_TIME = "2025-11-18_14-56-13"  # 2022-24 WPs + extra kl
# PARQUET_TIME = "2025-11-17_09-49-01"  # 2022-24 WPs + high stats -- USE THIS ONE
# PARQUET_TIME = "2025-11-18_13-48-48"  # 2022-24 WPs + high stats + 3XT + 4XT
# PARQUET_TIME = "2025-11-18_13-48-35"  # 2022-24 WPs + high stats + MHH

CURRENT_DIRPATH = str(Path().absolute())
VERSION = 'v18'
VARS = '22to24_bTagWPbatch3XT4XT'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

OPTIMIZE_SPACE = False
N_CLASSES = None
N_FOLDS = None

################################


def run_training(
    dataset_dirpath: str, output_dirpath: str,
    eos_dirpath: str=None, fold: int=None,
    batch: str='iterative', memory: str='10GB', queue: str="workday"
):
    OUTPUT_DIRPATH = os.path.join(output_dirpath, VERSION, VARS, CURRENT_TIME)
    if not os.path.exists(OUTPUT_DIRPATH):
        os.makedirs(OUTPUT_DIRPATH)

    # Dict defining which samples are in what classes (see `resolved_BDT.py` for more details)
    CLASS_SAMPLE_MAP = get_class_sample_map(dataset_dirpath)
    N_CLASSES = len(CLASS_SAMPLE_MAP)

    # Getting number of folds in the dataset
    N_FOLDS = get_n_folds(dataset_dirpath)

    # txt file pointing to location of standardized dataset used for training
    #  and therefore the default location for testing
    dataset_filepath = os.path.join(OUTPUT_DIRPATH, "dataset_filepath.txt")
    with open (dataset_filepath, "w") as f:
        f.write(dataset_dirpath)

    # Dict of hyperparameters for the model -- necessary to store for evaluation
    param_filepath = os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_best_params.json')
    if OPTIMIZE_SPACE:  # Need to change this to use condor as well..
        print('OPTIMIZING SPACE')
        
        param, num_trees = opt.optimize_hyperparams(
            dataset_dirpath, N_CLASSES, param_filepath, verbose=True
        )
    else:
        param, num_trees = opt.init_params(N_CLASSES)
    param['eval_metric'] = 'merror'
    with open(param_filepath, 'w') as f:
        json.dump(param, f)
    param = list(param.items()) + [('eval_metric', 'mlogloss')]

    evals_result_dict = {f"fold_{fold_idx}": dict() for fold_idx in range(N_FOLDS)}
    eval_result_filename = f'{CURRENT_TIME}_BDT_eval_result'

    # Train the model
    if batch == "iterative":
        for fold_idx in range(N_FOLDS):
            if fold is not None and fold != fold_idx: continue
            print(f"fold {fold_idx}")

            train_dm, val_dm, test_dm = get_train_DMatrices(dataset_dirpath, fold_idx)

            # Train bdt
            evallist = [(train_dm, 'train'), (val_dm, 'test'), (test_dm, 'val')]
            booster = xgb.train(
                param, train_dm, num_boost_round=num_trees, 
                evals=evallist, early_stopping_rounds=10, 
                verbose_eval=25, evals_result=evals_result_dict[f"fold_{fold_idx}"],
            )

            booster.save_model(os.path.join(OUTPUT_DIRPATH, f'{CURRENT_TIME}_BDT_fold{fold_idx}.model'))

            with open(os.path.join(OUTPUT_DIRPATH, f'{eval_result_filename}{fold_idx}.json'), 'w') as f:
                json.dump(evals_result_dict[f"fold_{fold_idx}"], f)
            
            # Print perf on test dataset
            print(booster.eval(test_dm, name='test', iteration=booster.best_iteration))
            print('='*100)
    elif batch == "condor":
        # Runs condor submission script
        submit(dataset_dirpath, OUTPUT_DIRPATH, eos_dirpath, N_FOLDS, memory=memory, queue=queue)

    # Merge the eval results dict into one dict
    for fold_idx in range(N_FOLDS):
        with open(os.path.join(OUTPUT_DIRPATH, f'{eval_result_filename}{fold_idx}.json'), 'r') as f:
            evals_result_dict[f"fold_{fold_idx}"] = json.load(f)
        subprocess.run(['rm', os.path.join(OUTPUT_DIRPATH, f'{eval_result_filename}{fold_idx}.json')], check=True)
    with open(os.path.join(OUTPUT_DIRPATH, f'{eval_result_filename}.json'), 'w') as f:
        json.dump(evals_result_dict, f)

if __name__ == "__main__":
    args = parser.parse_args()
    
    dataset_dirpath = os.path.join(args.dataset_dirpath, "")
    output_dirpath = os.path.join(args.output_dirpath, "")
    eos_dirpath = os.path.join(args.eos_dirpath, "")

    run_training(
        dataset_dirpath, output_dirpath, 
        eos_dirpath=eos_dirpath, fold=args.fold, 
        batch=args.batch, memory=args.memory, queue=args.queue
    )
