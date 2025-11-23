# Multiclass BDT model for $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis

## Introduction
This is a repository containing the code necessary to train and evaluate a Multiclass BDT event discrimintator for the CMS Run3 $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis. The model is trained on MC and evaluated on MC and Data. The structure of the repostory has gone through multiple iterations, but currently has the following main components each split into their own sub-folder: pre-processing, training, evaluation, plotting. Lets take a look at each of them in the order they are used.

The python environment necessary for this repository comes from HiggsDNA (explained below) as a conda enviroment. See the [HiggsDNA documentation](https://higgs-dna.readthedocs.io/en/latest/index.html) for instructions on how to install.


## Pre-processing
### HiggsDNA
The first step in any data analysis is the pre-processing of the data, which for us begins outside of this repo with HiggsDNA. All CMS Run3 $H \rightarrow \gamma\gamma$ analyses live under the Hgg PAG, and use the centralized [HiggsDNA](https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA) processing framework. This HiggsDNA framework is very robust and too complicated to be discussed in detail here, but I will summarize the main points. HiggsDNA is a [coffea](https://coffea-hep.readthedocs.io/en/latest/)-based framework that takes CMS NanoAOD root files and performs the initial pre-selection (HLT + basic object selections), skimming (removing unecessary columns of the dataset), and reconstruction (computing new useful variables). While the structure of HiggsDNA is customizable for any analysis (and therefore can be looked at as a template for coffea-based analyses), the framework is designed primarily for $H \rightarrow \gamma\gamma$ analyses. The output of HiggsDNA are `.parquet` files, and the code in this repository assumes the files used come from HiggsDNA (i.e. they are in the `.parquet` format and have the necessary variables).

### Preprocess
Assuming you have `.parquet` files from HiggsDNA, the first steo in this repository is to run a second pre-processing step (located under the `preprocessing` directory) that creates extra variables necessary for the Multiclass BDT training + evaluation. The reason for a second pre-processing step is purely logistical: the HiggsDNA framefork is centralized for all $H \rightarrow \gamma\gamma$ analyses and therefore is slow to update with code changes, whereas for ML training you want to have a quick turn-around for model development to be able to tweak the variables used. In essence, having a secondary pre-processing step independent of HiggsDNA allows us to de-couple the model development and optimization from HiggsDNA.

To run the pre-processing, use the `preprocess.py` script as follows:

`python preprocess.py <filepath_to_input_eras> --output_dirpath <path_to_output_directory>`

#### Structure of `input_eras` file
The `input_eras` file defines what eras of MC and Data to use for training and validation. Typically you will use as many MC eras as are available to get the best statistics during training, but it can be helpful to restrict the eras for debugging or testing new features.
```
# MC
/path/to/MC/era/1
/path/to/MC/era/2
# /path/to/unused/MC/era/3
# Data
/path/to/Data/era/1
```

The `input_eras` filepaths are expecting a directory whose immediate children are all the MC (Data) samples for that era. For example, the directory structure for the filepaths flags would look something like the following:
```bash
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
After running the `preprocess.py` script, all the necessary variables have been added to the input `.parquet` files, and you are almost ready to begin training. The final step is to run the `BDT_preprocessing.py` script in order to create lightweight training (and testing) files that have the minimal necessary variables for training and evaluating, with the proper variable standardization. The reason for making these lightweight files is two-fold: to save storage space on the cluster by not duplicating unecessary columns, and to ensure the training files are static. Saving storrage space is important because disk space is fundamentally a limited resource and it is easy (especially in ML contexts) to run out quickly. An added benefit of saving disk space with lightweight files, is that this makes parallel training of multiple models much easier because the required RAM is much smaller due to the lightweight files. Currently, this repositry doesn't make use of any parallelization (multi-processng, multi-threading, etc) but by setting things up the way we have, this would be an easy feature to implement. Keeping the training the training files static is important for an even more fundamental reason: backwards compatibility. If we always keep a copy of the data used to train a model -- and we separate the variable-dependent dataset creation from the model training code -- we can cross-check a model's training data, evaluate on new data, or even re-train a model at any time. This lets us quickly and easily create many different versions of our BDT model and we can compare the performance across versions *without changing any code*.

To run the variable standardization and training dataset creatin, use the `BDT_preprocessing.py` script as follows:

`python BDT_preprocessing.py <filepath_to_input_eras> <filepath_to_BDT_config_py_file> <filepath_to_dump_output_files>`

The `input_eras` file is the same as the one used for the `preprocess.py` file.

#### Structure of *BDT_config.py file
the `DATASET_TAG` variable is a short string that will be included in the name of the lightweight dataset and should be treated as a concise way of converying the most important properties of that dataset. I typically include the eras of the dataset and a short description of the variables that *differ from normal* -- what counts as differing from normal is up to you. It's important that you change this variable whenever you change the config file as this variable is your main way of tracking what is in a dataset.

The `CLASS_SAMPLE_MAP` variable exclusively defines the samples in the training datasets, as well as how to split the samples into distinct classes.
The structure of this map has the keys as the name of the classes, and the values as a list of wildcard sample-names (i.e. regex-like formatting) of that class.
```python
{
  'class 1': ['glob*name*1*', '*globname*2', etc]
}
```
The regex-like expressions are used to find the appropriate files for each of the classes and dataset types. The files are found by splitting the wildcard sample-names at the '*' characters, and selecting any samples that have *ALL* (with 1 exception) of the sub-strings. For example, the wildcard sample-name '*ttH*' matches to any filepath that contains the substring 'ttH'. 

The matching for finding samples to put into the datasets is done using the `re` module, so all the normal regex rules are available. The order of the different substrings matters, and the matching will look for them in the order they appear (although the distance between two matches does not matter). 
The exception to the *ALL* matching comes from the '!' symbol, which allows users to anti-match a substring. Anti-matching in this case means you can specify a substring that -- if matched to a filepath -- indicates you want to exclude that filepath. For example, the wildcard sample-name '*ttH*!Htobb*' indicates that you want to match all samples that contain 'ttH' but do *not* contain 'Htobb' afterwards.
After being used to find the samples, the found samples are then processed through this file into the lightweight training and testing files for the BDT. The wildcard sample-names are then used again any time users want to load the training or train-testing files for downstream tasks (I define train-testing to mean the test files of the train samples). These downstream loading tasks are done with the same function, which allows full use of regex tools. Leading and trailing '*' are therefore redundant.
The other necessary sets are the TRAIN_ONLY_SAMPLES and TEST_ONLY_SAMPLES. These sets define samples that should only be used for training or testing (as compared to the default behavior of the training files which is to be split into training and testing). The training-only samples should be contained in the CLASS_SAMPLE_MAP and should therefore be a subset of the training samples. There is a check that requires samples that match to TRAIN_ONLY_SAMPLES to also match to the CLASS_SAMPLE_MAP. The TEST_ONLY_SAMPLES, on the other hand, should be disjoint from the training, there is a check to ensure the samples *don't* match to CLASS_SAMPLE_MAP.

#### Structure of `--output_dirpath` directory:
```bash
<output_dirpath>
├── timestamp of training 1 dataset
    ├── MC/Data Era 1
    ├── MC/Data Era 2
    └── etc
└── timestamp of training 2 dataset
    ├── MC/Data Era 1
    ├── MC/Data Era 2
    └── etc
└── etc
```
If you choose to use the extra `--remake_test` flag (see below), the `--output_dirpath` flag should change from the general directory, to the specific timestamp directory associated with the test dataset you want to change. The names of the training dataset directories are timestamps of when the `BDT_preprocessing.py` file was run -- this means you do not know what is in the datasets by looking at them. **You must keep track of what eras, samples, and variables were used in the creation of a given dataset** I have done this using comments in the `run_training.py` file under the `training` directory, but you are free to keep track as you wish, just remember this information is not saved.

There are 4 extra flags: 
1. `--debug` prints out debug statements
2. `--dryrun` runs the code without actually saving out the training files (this can be helpful for debugging)
3. `--plots` makes plots of the BDT input variables before and after variable standardization (useful to understand what is going into the training)
4. `--remake_test` is a special flag for changing the test data stored in the output. This flag should *not* be used when you are creating an initial dataset for training. Rather, the `--remake_test` flag is to be used when you have already made a dataset for training and you want to change the samples in the test datasets (e.g. you have a new process you want to evlaute on, or there's a new EFT process to be evaluated, etc). This flag allows you to change the test files *without changing the training files* and using the proper variable standardization for that training. If you use this flag, the `--output_dirpath` needs to change from the directory that stores all the various training files to the directory of the specific training files for the training you want to use. See below for a visual example


## Training
Once you have run the `preprocess.py` and `BDT_preprocessing.py` scripts you are ready to train a model! Lucikly, because we setup the preprocessing and variable standardization in a backwards-compatible way (and split up each training dataset into its own directory), the training itself is extremely simple. You only need to run the `run_training.py` file (located under the `training` directory) with the `LPC_FILEPREFIX` variable changed to the location of your `--output_dirpath` from the `BDT_preprocessing.py` script, and the `PARQUET_TIME` variable set to the specific training dataset you would like to use. The `VERSION` and `VARS` variables are there only to help dilineate and organize the many versions of models you will train while optimizing your BDT.

To run the training, use the `run_training.py` script as follows:

`python run_training.py <path_to_dataset_dirpath>`

Extra arguments:
1. `--output_dirpath` changes the dump location fo the trained model files, by default the training files will be dumped under `HHtobbyy/MultiClassBDT_model_outputs/{VERSION}/{VARS}/{timestamp_of_training}`
2. `--eos_dirpath` changes the path for the eos space for intermediate files to be placed in, by default the path is `root://cmseos.fnal.gov//store/user/<username>/condor_train`, but this will obviously not work for people using LXPLUS or other Tier 1-3 clusters
3. `--optimize_space` will toggle on the hyperparameter optimization (this is very computationally expensive and should only be done *after* you've finalized your input variables and dataset eras and processes) -- if you would like to understand how the hyperparameters are optimized, you can read the `optimize_hyperparams.py` file and the corresponding blog-post its implemented from
4. `--fold` restricts the training to only perform one fold -- this is only important for batch jobs. If you are training a model with only 1 fold, you can specify that with N_FOLDS in the `BDT_preprocessing.py` script
5. `--batch` allows for selection of normal submission (`iterative`) or batch submission (currently only `condor`), see the `<batch_type>_training.py` files for details on how this is impemented
6. `--memory` changes the memory requested in the batch job
7. `--queue` changes the queue requested in the batch job


## Evaluation
Once you've trained a model, the last step is evaluating your test dataset with this model. Again, like with the training, things are easy because we put so much work into our dataset creation and management. All we need is the `evaluate_model.py` file (located under the `evaluation` directory),

To run the evaluation, use the `evaluate_model.py` file as follows:

`python evaluate_model.py <dirpath_to_trained_model> --dataset_dirpath <dirpath_to_training_dataset>`

The `training_dirpath` argument is the output directory of the training, while the `--dataset_dirpath` flag is optional -- it defaults to the same `--output_dirpath` from the `BDT_preprocessing.py` script used for the training specified in the `training_dirpath` flag.

Extra arguments:
1. `--dataset` runs the evluation of the train, test, train-test, or all datasets (defaults to test)
2. `--syst_name` runs the evaluation of the the nominal or all systematics for the specified datasets


## Plotting
The final action you may want to take is to make various plots concerning the performance of a model. These plots are important for comparing the performance across models, as well as because these plots are the visual tools you will need to explain to others (and yourself) how well your model works. There are many different types of plots (and not all of the scripts located under the `plotting` directory are working), but the structure of how to use them is fairly uniform and can be understood for each plotting script separately.

### ROCs
One of the most important plots (and the only one I will explain how to make here) are the Receiver Operator Characteristic (ROC) curves. The ROC curves describe how well a discriminator rejects background and accepts signal. For our analysis we break our Multiclass BDT output into 2 discriminators, one called DttH (designed to discriminate the $HH$ signal from the $ttH$ background) and the other called DQCD (designed to discriminate the $HH$ signal from the $H$ and QCD backgrounds). The `plot_roc.py` script creates the ROC curves for DttH and DQCD and prints out the signal efficiency at 1 (2) background-efficiency value(s) for DttH (DQCD).

To make the ROC curves, run the `plot_roc.py` script as follows:

`python plot_roc.py <dirpath_to_trained_model> --dataset_dirpath <dirpath_to_training_dataset>`

The two input arguments for the `plot_roc.py` file are the same as the inputs for the `evaluate_model.py` file. Currently, the plotting scripts are not all working, but this will change in the near future.
