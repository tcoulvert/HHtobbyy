# Multiclass BDT model for $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis

## Introduction
This is a repository containing the code necessary to train and evaluate a Multiclass BDT event discrimintator for the CMS Run3 $HH \rightarrow b\bar{b}\gamma\gamma$ Analysis. The model is trained on MC and evaluated on MC and Data. The structure of the repostory has gone through multiple iterations, but currently has the following main components each split into their own sub-folder: pre-processing, event-discrimnation (the bulk of the rpository), categorization, and FinalFit. Lets take a look at each of them in the order they are used.

The python environment necessary for this repository comes from HiggsDNA (explained below) as a conda enviroment. See the [HiggsDNA documentation](https://higgs-dna.readthedocs.io/en/latest/index.html) for instructions on how to install.

## TO-DO:
The framework is mostly settled, but there are a couple of useful things that should be implemented.
1. Fix the `event_discrimination/plotting` scripts to work with the current module-based HHbbyy repo
2. Add code to the `fitting/parquet_to_root.py` script to cross-check outputs from resolved BDT and boosted BDT and output separate `.ROOT` files for the various resolved and boosted categories for downstream fitting. This requires making use of the `AUX_hash` column, but it should be straightforward to implement.

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


## Event-Discrimination
After running the `preprocess.py` script, all the necessary variables have been added to the input `.parquet` files, and you are almost ready to begin training the event-discriminator. The event-discriminator module is broken up into a few different sub-modules to allow for readability and for the code to be model-agnostic – that is, any model architecutre can be implemented so long as it follows the required structure. The sub-modules of the event-discriminator are: DFDataset, Model, models, training, evaluation, and plotting. Let's take a look at each of these sub-modules in the order they are run in a typical analysis flow.

### DFDataset
The first step to build the event-discriminator is to create the `DFDataset`, which represents the lightweight training (and testing) files that have the minimal necessary variables for training and evaluation, with the proper variable standardization. The reason for making these lightweight files is two-fold: to save storage space on the cluster by not duplicating unecessary columns, and to ensure the training files are static. Saving storrage space is important because disk space is fundamentally a limited resource and it is easy (especially in ML contexts) to run out quickly. An added benefit of saving disk space with lightweight files, is that this makes parallel training of multiple models much easier because the required RAM is much smaller due to the lightweight files. Currently, this repositry doesn't make use of any parallelization (multi-processng, multi-threading, etc) but by setting things up the way we have, this would be an easy feature to implement. Keeping the training the training files static is important for an even more fundamental reason: backwards compatibility. If we always keep a copy of the data used to train a model -- and we separate the variable-dependent dataset creation from the model training code -- we can cross-check a model's training data, evaluate on new data, or even re-train a model at any time. This lets us quickly and easily create many different versions of our BDT model and we can compare the performance across versions *without changing any code*.

The `DFDataset.py` file defines the DFDataset object, which is configured using a dictionary (explained below).

Once the DFDataset object is defined, training datasets can be created using the code below:
```python
# Makes the train dataset for 1 fold
dfdataset.make_train(fold_index, filepaths **kwargs)
# Makes the train dataset for all folds
dfdataset.make_all_train(filepaths, **kwargs)
```
The `filepaths` arg is a list of filepaths to use, which will be converted into a dictionary where the filepaths are gouped by class (see DFDataset config). The kwargs are optional arguments to be passed to the multifold wrapper – located at `workspace_utils/retrieval_utils/multifold`, allowing code to run across multiple folds in parallel. 

Similarly, testing datasets can be created using the code below:
```python
# Makes the test dataset for 1 fold
dfdataset.make_test(fold_index, filepaths, force=(True)False, **kwargs)
# Makes the test dataset for all folds
dfdataset.make_all_test(filepaths, force=(True)False, **kwargs)
```
The `force` flag allows you to re-run the test dataset creation and override the current files, by default it `force=False`, so only new test datasets will be created. The testing dataset for each fold must be created *after* the train dataset for that fold, because the train dataset defines the standardization values that need to be applied. However, each fold can be run independently, just like for the training dataset.

#### Structure of DFDataset config
The config is required to have certain keys which are fundamental to the creation and running of a new model, however you can optionally pass more keys that will override the default values for the DFDataset member variables. The full list of member variables is given in the DFDataset class, along with comments explaining their purpose, but I will go over the required variables below:

```python
{
    # Path (can include XRD redirectors for EOS) for output directory for DFDataset files
    "output_dirpath": "path/to/output/directory/for/DFDataset",

    # Short description of the dataset
    #   -> no required structure for the tag, it's simply for your own house-keeping
    "dataset_tag": "tagline_of_dataset",

    # List of training variables
    "model_vars": ["var1", "var2", ...],

    # List of auxiliary variables
    #   -> Unstandardized variables to be used downstream in the analysis, including (unstandardized) duplicates from the training list which will be renamed using the 'aux_var_prefix' member variable of DFDataset

    # Map from processes (file_regexs) to class groupings
    #   -> BE CAREFUL!! If your regexs are poorly chosen your model *will* have different class groupings than you expect there is some code to check for this using `workspace_utils/retrieval_utils/check_train_filepaths`, but that only happens if you get your filepaths
    #   -> implicitly defines number of classes
    "class_sample_map": {
        "class1": ["file_regex1", "file_regex2"],
        "class2": ["file_regex3", "file_regex4"],
        ...
    }
}
```

#### Structure of DFDataset in the `output_dirpath` directory:
```bash
<output_dirpath>
├── <dataset_tag of training dataset 1>_<timestamp of training dataset 1>
    ├── MC/Data Era 1
    ├── MC/Data Era 2
    └── etc
└── <dataset_tag of training dataset 2>_<timestamp of training dataset 2>
    ├── MC/Data Era 1
    ├── MC/Data Era 2
    └── etc
└── etc
```
The DFDataset structure follows from the general output structure of HiggsDNA (see pre-processing).

### Model
Now that the dataset is defined, we are ready to define the model used for training. In order to be agnostic to model architecture, the model definition is split into two modules: `Model` (this section) and `models` (next section). `Model` defines the basic structure any model must have in order to operate in this framework via abstract classes. Therefore, the code in `Model` is largely a template of how the `HHtobbyy` module expects ML models to be constructed, with the execption of some (small) class methods that exist across specific ML implementations. In contrast, the `models` module defines the specific implementation for each architecture implemented.

In practice, the rest of the `HHtobbyy` code can simply interact with the `Model` module and agnostically call any ML architecture. The configuration, training, and evaluation are all encompassed in the `Model`, so that downstream tasks can call a uniform API for any model.

As a user, the `Model` vs `models` difference is mostly just an abstraction and can be ignored. However, if you want to implement your own model, you must do-so following the template of the `Model` module and do-so inside the `models` module.

### models
As stated above, the `models` module contains the implementations for each of the architectures desired for the event-discriminator. The reason why this is nice, is it allows you to try multiple different architectures (e.g. BDT and DNN) and easily compare the performance between them without needing to re-write any code. This is an extremely common scenario in ML, and this split `Model` vs. `models` setup allows for you to write, rewrite, or completely change the architecture while having both a structure to follow and the freedom to use any model.

Currently, there are 2 models implemented: `MLP` and `XGBoostBDT`. `MLP` is a (simple) PyTorch-based Multiclass-DNN, while `XGBoostBDT` is a XGBoost-based Multiclass-BDT. I won't discuss here the details of each model, rather I invite you to read the code of each if you're interested.

### training
The bulk of the training code is implemented on a per-model basis in the `models` module. However, there are a few utility functions (as well as Condor submission) that are contained in the `training` module.

### evaluation
Similarly as above, the bulk of the evaluation code is implemented on a per-model basis in the `models` module. However, there are a few utility functions that are contained in the `evaluation` module.

### plotting
The final action you may want to take is to make various plots concerning the performance of a model. These plots are important for comparing the performance across models, as well as because these plots are the visual tools you will need to explain to others (and yourself) how well your model works. There are many different types of plots, but currently the plotting scripts aren't implemented for the module-based `HHtobbyy`. They will be added soon.
