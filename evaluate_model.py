# Stdlib packages
import copy
import glob
import os
import re
import warnings

# Common Py packages
import awkward as ak
import numpy as np

# HEP packages
import xgboost as xgb

# ML packages
from sklearn.metrics import log_loss

# Module packages
from data_processing_BDT import process_data


# lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v3/"
lpc_fileprefix = "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v3.1/"
Run3_2022 = 'Run3_2022_mergedFullResolved/sim'
Run3_2023 = 'Run3_2023_mergedFullResolved/sim'
Run3_2024 = 'Run3_2024_mergedFullResolved/sim'

def get_filepath_dict(syst_name: str='nominal'):
    return {
        'ggF HH': [
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", # central v2 preEE name
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",  # central v2 postEE name
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",  # thomas name
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",

            # lpc_fileprefix+Run3_2022+f"/preEE/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet", 
            # lpc_fileprefix+Run3_2022+f"/postEE/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet",
            # lpc_fileprefix+Run3_2023+f"/preBPix/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet", 
            # lpc_fileprefix+Run3_2023+f"/postBPix/VBFHHto2B2G_CV_1_C2V_1_C3_1/{syst_name}/*merged.parquet",

            # kappa lambda scan #
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2022+f"/preEE/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-0p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-2p45_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGlutoHH_kl-5p00_kt-1p00_c2-0p00/{syst_name}/*merged.parquet",
        ],
        'ttH + bbH': [
            # ttH
            lpc_fileprefix+Run3_2022+f"/preEE/ttHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/ttHToGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/ttHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/ttHtoGG/{syst_name}/*merged.parquet",
            # bbH
            lpc_fileprefix+Run3_2022+f"/preEE/bbHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/bbHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/bbHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/bbHtoGG/{syst_name}/*merged.parquet",
        ],
        'VH': [
            # VH
            lpc_fileprefix+Run3_2022+f"/preEE/VHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/VHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/VHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/VHtoGG/{syst_name}/*merged.parquet",
            # ZH
            lpc_fileprefix+Run3_2022+f"/preEE/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/ZH_Hto2G_Zto2Q_M-125/{syst_name}/*merged.parquet",
            # W-H
            lpc_fileprefix+Run3_2022+f"/preEE/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/WminusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            # W+H
            lpc_fileprefix+Run3_2022+f"/preEE/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/WplusH_Hto2G_Wto2Q_M-125/{syst_name}/*merged.parquet",
        ],
        'non-res + ggFH + VBFH': [
            # GG + 3Jets 40-80
            lpc_fileprefix+Run3_2022+f"/preEE/GGJets_MGG-40to80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GGJets_MGG-40to80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GGJets_MGG-40to80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GGJets_MGG-40to80/{syst_name}/*merged.parquet",
            # GG + 3Jets 80-
            lpc_fileprefix+Run3_2022+f"/preEE/GGJets_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GGJets_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GGJets_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GGJets_MGG-80/{syst_name}/*merged.parquet",
            # GJet pT 20-40
            lpc_fileprefix+Run3_2022+f"/preEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GJet_PT-20to40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            # GJet pT 40-inf
            lpc_fileprefix+Run3_2022+f"/preEE/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GJet_PT-40_DoubleEMEnriched_MGG-80/{syst_name}/*merged.parquet",
            # ggF H
            lpc_fileprefix+Run3_2022+f"/preEE/GluGluHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/GluGluHtoGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/GluGluHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/GluGluHtoGG/{syst_name}/*merged.parquet",
            # VBF H
            lpc_fileprefix+Run3_2022+f"/preEE/VBFHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2022+f"/postEE/VBFHToGG/{syst_name}/*merged.parquet",
            lpc_fileprefix+Run3_2023+f"/preBPix/VBFHtoGG/{syst_name}/*merged.parquet", 
            lpc_fileprefix+Run3_2023+f"/postBPix/VBFHtoGG/{syst_name}/*merged.parquet",
        ],
    }

MODEL_FILEPATH = os.path.join(
    '/uscms/home/tsievert/nobackup/XHYbbgg/HHtobbyy/MultiClassBDT_model_outputs/v14/v3_vars_DijetMass_22_23/2025-07-08_15-47-07',
    ''
)
MOD_VALS = (5, 5)
FORCE_REEVAL = False

order = ['ggF HH', 'ttH + bbH', 'VH', 'non-res + ggFH + VBFH']

# Booster parameters #
param = {}
# v12 #
param['eta']              = 0.05 # learning rate
num_trees = round(25 / param['eta'])  # number of trees to make
param['max_depth']        = 10  # maximum depth of a tree
param['subsample']        = 0.2 # fraction of events to train tree on
param['colsample_bytree'] = 0.6 # fraction of features to train tree on
param['num_class']        = len(order) # num classes for multi-class training
param['device']           = 'cuda'
param['tree_method']      = 'gpu_hist'
param['max_bin']          = 512
param['grow_policy']      = 'lossguide'
param['sampling_method']  = 'gradient_based'
param['min_child_weight'] = 0.25
# Learning task parameters
param['objective']   = 'multi:softprob'   # objective function
param['eval_metric'] = 'merror'
param = list(param.items()) + [('eval_metric', 'mlogloss')]  # evaluation metric for cross validation

# # custom eval_metrics
# def one_hot_encoding(cat_labels: np.ndarray):
#     one_hot = np.zeros((np.shape(cat_labels)[0], np.max(cat_labels)+1))
#     for i in range(np.max(cat_labels)):
#         one_hot[:, i] = (cat_labels == i)
#     return one_hot

# def mlogloss_binlogloss(
#     predt: np.ndarray, dtrain: xgb.DMatrix, mLL=True, **kwargs
# ):
#     assert (len(kwargs) == 0 and mLL) or len(kwargs) == (len(order) - 1)

#     mweight = dtrain.get_weight()
#     monehot = one_hot_encoding(dtrain.get_label())
#     mlogloss = log_loss(monehot, predt, sample_weight=mweight, normalize=False)

#     bkgloglosses = {}
#     for i, (key, value) in enumerate(kwargs.items(), start=1):
#         bkgbool = np.logical_or(mweight == 0, mweight == i)
#         bkgloglosses[key] = value * log_loss(
#             monehot[bkgbool], predt[bkgbool, 0] / (predt[bkgbool, 0] + predt[bkgbool, i]),
#             sample_weight=mweight[bkgbool], normalize=False
#         )

#     if len(bkgloglosses) > 0 and mLL:
#         return f'mLL+binLL@{bkgloglosses.keys()}', float(np.sum([mlogloss]+list(bkgloglosses.values())))
#     elif len(bkgloglosses) > 0:
#         return f'binLL@{bkgloglosses.keys()}', float(np.sum(bkgloglosses.values()))
#     else:
#         return 'mLL', float(mlogloss)

def thresholded_weighted_merror(predt: np.ndarray, dtrain: xgb.DMatrix, threshold=0.95):
    """Used when there's no custom objective."""
    # No need to do transform, XGBoost handles it internally.
    weights = dtrain.get_weight()
    thresh_weight_merror = np.where(
        np.logical_and(
            np.max(predt, axis=1) >= threshold,
            np.argmax(predt, axis=1) == dtrain.get_label()
        ),
        0,
        weights
    )
    return f'WeightedMError@{threshold:.2f}', np.sum(thresh_weight_merror)

# load and pre-process the data
VARIATIONS_FILEPATHS_DICT = {
    syst_name: get_filepath_dict(syst_name=syst_name) for syst_name in [
        'nominal',
        'ScaleEB2G_IJazZ_up', 'ScaleEB2G_IJazZ_down', 'ScaleEE2G_IJazZ_up', 'ScaleEE2G_IJazZ_down',
        'Smearing2G_IJazZ_up', 'Smearing2G_IJazZ_down',
        'jec_syst_Total_up', 'jec_syst_Total_down', 'jer_syst_up', 'jer_syst_down',
        'FNUF_up', 'FNUF_down', 'Material_up', 'Material_down'
    ]
}

# load and pre-process the data
DATA_FILEPATHS_DICT = {
    'Data': [
        # 2022
        lpc_fileprefix+Run3_2022[:-4]+f"/data/DataC_2022/*merged.parquet",
        lpc_fileprefix+Run3_2022[:-4]+f"/data/DataD_2022/*merged.parquet",
        lpc_fileprefix+Run3_2022[:-4]+f"/data/Data_EraE/*merged.parquet",
        lpc_fileprefix+Run3_2022[:-4]+f"/data/Data_EraF/*merged.parquet",
        lpc_fileprefix+Run3_2022[:-4]+f"/data/Data_EraG/*merged.parquet",
        # 2023
        lpc_fileprefix+Run3_2023[:-4]+"/data/Data_EraC/*merged.parquet",
        lpc_fileprefix+Run3_2023[:-4]+"/data/Data_EraD/*merged.parquet",
    ],
}

# Sorts the predictions to map the output to the correct event
def sorted_preds(preds, data_aux, sample, sorted_preds=False, sort_variable='hash'):

    if np.size(np.unique(sample[sort_variable])) != np.size(sample[sort_variable]):
        raise Exception(f"The sort_variable you chose ({sort_variable}) does not have uniquely defined values for all events. This will cause sorting failures.")
    
    if not sorted_preds:
        flat_preds = np.concatenate([preds[fold_idx] for fold_idx in range(len(data_aux))])
        preds_sort = np.argsort(
            np.concatenate([data_aux[f"fold_{fold_idx}"].loc[:, sort_variable].to_numpy() for fold_idx in range(len(data_aux))])
        )
    else:
        flat_preds = preds
        preds_sort = np.arange(len(flat_preds))

    sample_sort = np.argsort(np.argsort(
        ak.to_numpy(sample[sort_variable], allow_missing=False)
    ))

    if not sorted_preds and np.any(
        np.concatenate(
            [data_aux[f"fold_{fold_idx}"].loc[:, sort_variable].to_numpy() for fold_idx in range(len(data_aux))]
        )[preds_sort][sample_sort] != ak.to_numpy(sample[sort_variable], allow_missing=False)
    ):
        raise Exception(f"Sort failed.")

    return flat_preds[preds_sort][sample_sort]

def get_file_sample_name(dirpath: str, variation: str):
    end_idx = dirpath.find(variation) - 1
    start_idx = dirpath[:end_idx].rfind('/') + 1

    if end_idx == -2 or start_idx == 0: return ''

    return dirpath[start_idx:end_idx]

def get_ttH_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])

def get_QCD_score(multibdt_output):
    return multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])

CATS = [
    [0.987, 0.9982],
    [0.92, 0.994],
    [0.92, 0.9864],
]

def pass_category(multibdt_output, cat_i):
    pass_ttH = get_ttH_score(multibdt_output) > CATS[cat_i][0]
    pass_QCD = get_QCD_score(multibdt_output) > CATS[cat_i][1]

    pass_prevQCD = get_QCD_score(multibdt_output) <= (
        CATS[cat_i-1][1] if cat_i > 1 else 1
    )

    return pass_ttH & pass_QCD & pass_prevQCD


dirpath_addition = '_MultiBDT_output'
filename_addition = '_MultiBDT_output'
basefilename_postfix = 'Run3_202x_mergedFullResolved'  # Resolved

jet_prefix = 'nonResReg_DNNpair' if re.search('MbbRegDNNPair', MODEL_FILEPATH) is not None else (
    'nonResReg' if re.search('MbbReg', MODEL_FILEPATH) is not None else 'nonRes'
)

## MC SAMPLES ##
# Load parquet files #
for i, sample_name in enumerate(order):

    for variation, variation_filepath_dict in VARIATIONS_FILEPATHS_DICT.items():

        for dirpath in variation_filepath_dict[sample_name]:

            if variation != 'nominal' and re.search('H', get_file_sample_name(dirpath, variation).upper()) is None: continue
            if re.search('Q', get_file_sample_name(dirpath, variation).upper()) is not None: continue

            print('======================== started \n', dirpath)

            for parquet_filepath in glob.glob(dirpath):
                dest_filepath = parquet_filepath[:parquet_filepath.find('Run3_202')+len(basefilename_postfix)] + dirpath_addition + parquet_filepath[parquet_filepath.find('Run3_202')+len(basefilename_postfix):parquet_filepath.rfind('.')] + filename_addition + parquet_filepath[parquet_filepath.rfind('.'):]
                if not os.path.exists(dest_filepath[:dest_filepath.rfind('/')]):
                    os.makedirs(dest_filepath[:dest_filepath.rfind('/')])
                elif not FORCE_REEVAL and os.path.exists(dest_filepath):
                    print(f'file already exists at \n{dest_filepath}\n{"="*60}')
                    continue

                sample = ak.from_parquet(parquet_filepath)
                sample = sample[
                    sample[f'{jet_prefix}_has_two_btagged_jets']
                    & sample['fiducialGeometricFlag']
                    & (
                        (sample['lead_mvaID'] > -0.7)
                        & (sample['sublead_mvaID'] > -0.7)
                    )
                ]

                (
                    NOTHING_IGNORE,
                    IGNORE_data_df_dict, SAMPLE_data_test_df_dict, 
                    IGNORE_data_hlf_dict, IGNORE_label_dict,
                    SAMPLE_data_hlf_test_dict, SAMPLE_label_test_dict, 
                    SAMPLE_hlf_vars_columns_dict,
                    IGNORE_data_aux_dict, SAMPLE_data_test_aux_dict
                ) = process_data(
                    {"sample": [parquet_filepath]}, MODEL_FILEPATH, order=['sample'], mod_vals=MOD_VALS, k_fold_test=True,
                    save=False, std_json_dirpath=MODEL_FILEPATH, 
                    jet_prefix=jet_prefix
                )

                sample_preds = []
                for fold_idx in range(len(SAMPLE_data_test_df_dict)):
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        booster = xgb.Booster(param)
                        booster.load_model(os.path.join(MODEL_FILEPATH, f'{MODEL_FILEPATH.split('/')[-1]}_BDT_fold{fold_idx}.model'))

                        bdt_test_sample_dict = xgb.DMatrix(
                            data=SAMPLE_data_hlf_test_dict[f"fold_{fold_idx}"], label=SAMPLE_label_test_dict[f"fold_{fold_idx}"], 
                            missing=-999.0, feature_names=list(SAMPLE_hlf_vars_columns_dict[f"fold_{fold_idx}"])
                        )

                        sample_preds.append(
                            booster.predict(
                                bdt_test_sample_dict, 
                                iteration_range=(0, booster.best_iteration+1)
                            )
                        )

                sample['MultiBDT_output'] = sorted_preds(
                    sample_preds, SAMPLE_data_test_aux_dict, sample
                )

                merged_parquet = ak.to_parquet(sample, dest_filepath)
                del sample
                print('======================== finished \n', dest_filepath)

## DATA ##
for dirpath in DATA_FILEPATHS_DICT['Data']:

    print('======================== started \n', dirpath)
    parquet_filepath = glob.glob(dirpath)[0]
    dest_filepath = parquet_filepath[:parquet_filepath.find('Run3_202')+len(basefilename_postfix)] + dirpath_addition + parquet_filepath[parquet_filepath.find('Run3_202')+len(basefilename_postfix):parquet_filepath.rfind('.')] + filename_addition + parquet_filepath[parquet_filepath.rfind('.'):]
    if not os.path.exists(dest_filepath[:dest_filepath.rfind('/')]):
        os.makedirs(dest_filepath[:dest_filepath.rfind('/')])
    elif not FORCE_REEVAL and os.path.exists(dest_filepath):
        print(f'file already exists at \n{dest_filepath}\n{"="*60}')
        continue
    
    data_sample = ak.from_parquet(parquet_filepath)
    data_sample = data_sample[
        data_sample[f'{jet_prefix}_has_two_btagged_jets']
        & data_sample['pass_fiducial_geometric']
        & (
            (
                data_sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90']
                & data_sample['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95']
            ) 
            if 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90' in data_sample.fields else (data_sample['mass'] > 0)
        ) & (
            (data_sample['lead_mvaID'] > -0.7)
            & (data_sample['sublead_mvaID'] > -0.7)
        )
    ]

    (
        NOTHING_IGNORE,
        DATA_data_df_dict, DATA_data_test_df_dict, 
        DATA_data_hlf_dict, DATA_label_dict,
        DATA_data_hlf_test_dict, DATA_label_test_dict, 
        DATA_hlf_vars_columns_dict,
        DATA_data_aux_dict, DATA_data_test_aux_dict
    ) = process_data(
        {"sample": [parquet_filepath]}, MODEL_FILEPATH, order=['sample'], mod_vals=MOD_VALS, k_fold_test=True,
        save=False, std_json_dirpath=MODEL_FILEPATH,
        jet_prefix=jet_prefix
    )

    bdt_train_data_dict = xgb.DMatrix(
        data=DATA_data_hlf_dict[f"fold_0"], label=DATA_label_dict[f"fold_0"], 
        missing=-999.0, feature_names=list(DATA_hlf_vars_columns_dict[f"fold_0"])
    )
    bdt_test_data_dict = xgb.DMatrix(
        data=DATA_data_hlf_test_dict[f"fold_0"], label=DATA_label_test_dict[f"fold_0"], 
        missing=-999.0, feature_names=list(DATA_hlf_vars_columns_dict[f"fold_0"])
    )

    test_preds = []
    for fold_idx in range(len(DATA_label_test_dict)):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            booster = xgb.Booster(param)
            booster.load_model(os.path.join(MODEL_FILEPATH, f'{MODEL_FILEPATH.split('/')[-1]}_BDT_fold{fold_idx}.model'))

            # all-fold eval
            BDT_train_preds = booster.predict(
                bdt_train_data_dict, 
                iteration_range=(0, booster.best_iteration+1)
            )
            BDT_test_preds = booster.predict(
                bdt_test_data_dict, 
                iteration_range=(0, booster.best_iteration+1)
            )

            BDT_all_preds = np.concatenate([BDT_train_preds, BDT_test_preds])
            BDT_all_preds = BDT_all_preds[
                np.argsort(
                    np.concatenate([DATA_data_aux_dict[f"fold_0"].loc[:, 'hash'].to_numpy(), DATA_data_test_aux_dict[f"fold_0"].loc[:, 'hash'].to_numpy()])
                )
            ]

            if fold_idx == 0:
                data_preds = copy.deepcopy(BDT_all_preds)
            else:
                data_preds += BDT_all_preds

                if fold_idx == len(DATA_label_test_dict) - 1:
                    data_preds = data_preds / len(DATA_label_test_dict)


            # single-fold eval
            bdt_test_data_fold = xgb.DMatrix(
                data=DATA_data_hlf_test_dict[f"fold_{fold_idx}"], label=DATA_label_test_dict[f"fold_{fold_idx}"], 
                missing=-999.0, feature_names=list(DATA_hlf_vars_columns_dict[f"fold_{fold_idx}"])
            )

            test_preds.append(
                booster.predict(
                    bdt_test_data_fold,
                    iteration_range=(0, booster.best_iteration+1)
                )
            )

    data_sample['MultiBDT_output'] = sorted_preds(
        data_preds, DATA_data_test_aux_dict, data_sample,
        sorted_preds=True
    )
    data_sample['MultiBDT_output_mod5'] = sorted_preds(
        test_preds, DATA_data_test_aux_dict, data_sample
    )

    # print(f"Total number of events in {dest_filepath}\n = {ak.num(data_sample, axis=0)}")
    # for cat_i, cat in enumerate(CATS):
    #     print(f"Passing number of events in cat{cat_i} {dest_filepath}\n = {ak.sum(pass_category(data_sample['MultiBDT_output'], cat_i), axis=0)}")

    merged_parquet = ak.to_parquet(data_sample, dest_filepath)
    del data_sample
    print('======================== finished \n', dest_filepath)
