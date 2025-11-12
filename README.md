# Multiclass BDT model for $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis


## Introduction
This is a repository containing the code necessary to train and evaluate a Multiclass BDT event discrimintator for the CMS Run3 $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis. The model is trained on MC and evaluated on MC and Data. The structure of the repostory has gone through multiple iterations, but currently has the following main components each split into their own sub-folder: pre-processing, training, evaluation, plotting. Lets take a look at each of them in the order they are used.


## Pre-processing
### HiggsDNA
The first step in any data analysis is the pre-processing of the data, which for us begins outside of this repo with HiggsDNA. All CMS Run3 $H \rightarrow \gamma\gamma$ analyses live under the Hgg PAG, and use the centralized [HiggsDNA](https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA) processing framework. This HiggsDNA framework is very robust and too complicated to be discussed in detail here, but I will summarize the main points. HiggsDNA is a [coffea](https://coffea-hep.readthedocs.io/en/latest/)-based framework that takes CMS NanoAOD root files and performs the initial pre-selection (HLT + basic object selections), skimming (removing unecessary columns of the dataset), and reconstruction (computing new useful variables). While the structure of HiggsDNA is customizable for any analysis (and therefore can be looked at as a template for coffea-based analyses), the framework is designed primarily for $H \rightarrow \gamma\gamma$ analyses. The output of HiggsDNA are `.parquet` files, and the code in this repository assumes the files used come from HiggsDNA (i.e. they are in the `.parquet` format and have the necessary variables).

### Preprocess
Assuming you have `.parquet` files from HiggsDNA, the first steo in this repository is to run a second pre-processing step (located under the `preprocessing` directory) that creates extra variables necessary for the Multiclass BDT training + evaluation. The reason for a second pre-processing step is purely logistical: the HiggsDNA framefork is centralized for all $H \rightarrow \gamma\gamma$ analyses and therefore is slow to update with code changes, whereas for ML training you want to have a quick turn-around for model development to be able to tweak the variables used. In essence, having a secondary pre-processing step independent of HiggsDNA allows us to de-couple the model development and optimization from HiggsDNA.

To run the pre-processing, use the `preprocess.py` script as follows:
>`python preprocess.py --sim_era_filepaths <filepath_for_MC_era_1>,<filepath_for_MC_era_2>,etc --data_era_filepaths <filepath_for_Data_era_1>,<filepath_for_Data_era_2>,etc --output_dirpath <path_to_output_directory>`
The `--sim_era_filepaths` (`--data_era_filepaths`) flag is expecting a directory whose immediate children are all the MC (Data) samples for that era. For example, the directory structure for the filepaths flags would look something like the following:
    ```
    MC/Data era 1
    ├── process 1
        ├── nominal
        ├── systematic variation 1 Up
        ├── systematic variation 1 Down
        ├── systematic variation 2 Up
        └── etc
    └── process 2
        ├── nominal
        ├── systematic variation 1 Up
        ├── systematic variation 1 Down
        ├── systematic variation 2 Up
        └── etc
    └── etc
    ```
By default the `preprocess.py` file will create a new `.parquet` file adjacent to each input file.

### Variable Standardization
After running the `preprocess.py` script, all the necessary variables have been added to the input `.parquet` files, and you are almost ready to begin training. The final step is to run the `resolved_BDT.py` script in order to create lightweight training (and testing) files that have the minimal necessary variables for training and evaluating, with the proper variable standardization. The reason for making these lightweight files is two-fold: to save storage space on the cluster by not duplicating unecessary columns, and to ensure the training files are static. Saving storrage space is important because disk space is fundamentally a limited resource and it is easy (especially in ML contexts) to run out quickly. An added benefit of saving disk space with lightweight files, is that this makes parallel training of multiple models much easier because the required RAM is much smaller due to the lightweight files. Currently, this repositry doesn't make use of any parallelization (multi-processng, multi-threading, etc) but by setting things up the way we have, this would be an easy feature to implement. Keeping the training the training files static is important for an even more fundamental reason: backwards compatibility. If we always keep a copy of the data used to train a model -- and we separate the variable-dependent dataset creation from the model training code -- we can cross-check a model's training data, evaluate on new data, or even re-train a model at any time. This lets us quickly and easily create many different versions of our BDT model and we can compare the performance across versions *without changing any code*.

To run the variable standardization and training dataset creatin, use the `resolved_BDT.py` script as follows:
>`python resolved_BDT.py --input_filepaths <filepath_to_JSON_dictionary_of_filepaths> --output_dirpath <filepath_to_dump_training_files>`
Structure of JSON dictionary for `--input_filepaths` flag:
    ```
    {
        'train-test': ['<filepath_to_process_1>', '<filepath_to_process_2>', etc],
        'train': ['<filepath_to_process_1>', '<filepath_to_process_2>', etc],
        'test': ['<filepath_to_process_1>', '<filepath_to_process_2>', etc]
    }
    ```
The structure of the JSON dictionary is meant to reflect the differing treatment of processes depending on the process and analysis: `'train-test'` refers to processes that will be used for training *and* testing (this is the majority of processes), `'train'` refers to processes used *only* for training (this is for extra samples to e.g. add statistics for training, but they could be private production and therefore not fit for evaluation), `'test'` refers to processes used *only* for testing (this is for samples to e.g. do EFT evaluations, or Data samples).

Structure of `--output_dirpath` directory:
    ```
    <output_dirpath>
    ├── timestamp for training 1 dataset
        ├── MC/Data Era 1
        ├── MC/Data Era 2
        └── etc
    └── timestamp for training 2 dataset
        ├── MC/Data Era 1
        ├── MC/Data Era 2
        └── etc
    └── etc
    ```
If you choose to use the extra `--remake_test` flag (see below), the `--output_dirpath` flag should change from the general directory, to the specific timestamp directory associated with the test dataset you want to change. The names of the training dataset directories are timestamps of when the `reasolved_BDT.py` file was run -- this means you do not know what is in the datasets by looking at them. **You must keep track of what eras, samples, and variables were used in the creation of a given dataset** I have done this using comments in the `run_training.py` file under the `training` directory, but you are free to keep track as you wish, just remember this information is not saved.

There are 4 extra flags: 
    1. `--debug` prints out debug statements
    2. `--dryrun` runs the code without actually saving out the training files (this can be helpful for debugging)
    3. `--plots` makes plots of the BDT input variables before and after variable standardization (useful to understand what is going into the training)
    4. `--remake_test` is a special flag for changing the test data stored in the output. This flag should *not* be used when you are creating an initial dataset for training. Rather, the `--remake_test` flag is to be used when you have already made a dataset for training and you want to change the samples in the test datasets (e.g. you have a new process you want to evlaute on, or there's a new EFT process to be evaluated, etc). This flag allows you to change the test files *without changing the training files* and using the proper variable standardization for that training. If you use this flag, the `--output_dirpath` needs to change from the directory that stores all the various training files to the directory of the specific training files for the training you want to use. See below for a visual example


## Training
Once you have run the `preprocess.py` and `resolved_BDT.py` scripts you are ready to train a model! Lucikly, because we setup the preprocessing and variable standardization in a backwards-compatible way (and split up each training dataset into its own directory), the training itself is extremely simple. You only need to run the `run_training.py` file with the `LPC_FILEPREFIX` variable changed to the location of your `--output_dirpath` from the `resolved_BDT.py` script, and the `PARQUET_TIME` variable set to the specific training dataset you would like to use. The `VERSION` and `VARS` variables are there only to help dilineate and organize the many versions of models you will train while optimizing your BDT.

To run the training, use the `run_training.py` script as follows:
>`python run_training.py`
By default the training files will be dumped under `HHtobbyy/MultiClassBDT_model_outputs/{VERSION}/{VARS}/{timestamp_of_training}`.

If you would like to also optimize the hyperparameters of your BDT (this is very computationally expensive and should only be done *after* you've finalized your input variables and dataset eras and processes), simply set the `OPTIMIZE_SPACE = True` boolean in the `run_training.py` file. This will call the `optimize_hyperparams.py` file, but everything is handled under-the-hood. If you would like to understand how the hyperparameters are optimized, you can read that file and the corresponding blog-post its implemented from.


## Evaluation
