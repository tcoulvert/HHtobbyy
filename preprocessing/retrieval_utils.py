# %matplotlib widget
# Common Py packages
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# HEP packages
import xgboost as xgb

# ML packages
from sklearn.model_selection import train_test_split

################################


RES_BKG_RESCALE = 1000  # 1e3
DF_SHUFFLE = True
RNG_SEED = 21
FILL_VALUE = -999

################################


def get_labelND(label1D):
    """
    Returns the ND label vector (one-hot encoded) from the 1D label vector (integer encoded).
    """
    return np.tile(np.arange(np.max(label1D)), (np.size(label1D), 1)) == label1D
def get_label1D(labelND):
    """
    Returns the 1D label vector (integer encoded) from the ND label vector (one-hot encoded).
    """
    return np.nonzero(labelND).flatten()

def get_Dataframe(filepath: str, aux: bool=False):
    schema = pq.read_schema(filepath)
    vars = [var for var in schema.names if ('AUX_' not in var) ^ aux]

    df = pq.read_table(filepath, columns=vars).to_pandas()

    return df
def get_Dataframes(get_filepaths, fold_idx: int, dataset: str):
    filepaths = get_filepaths(fold_idx, dataset)

    df, aux = None, None
    for i, bdt_class in enumerate(filepaths.keys()):
        class_df = pd.concat(
            [get_Dataframe(filepath) for filepath in filepaths[bdt_class]], ignore_index=True
        )
        class_aux = pd.concat(
            [get_Dataframe(filepath, aux=True) for filepath in filepaths[bdt_class]], ignore_index=True
        )
        class_aux['AUX_label1D'] = i
        class_aux['AUX_eventWeightTrain'] = class_aux['AUX_eventWeight']

        mask_field = [field for field in class_aux.columns if 'resolved_BDT_mask' in field][0]
        mask = (class_aux[mask_field] > 0)
        class_df = class_df.loc[mask]
        class_aux = class_aux.loc[mask]
        
        if df is None:
            df = class_df
            aux = class_aux
        else:
            df = pd.concat([df, class_df], ignore_index=True)
            aux = pd.concat([aux, class_aux], ignore_index=True)

    # Upweight resonant background and signal samples for training #
    # Non-Resonant background #
    nonres_mask = aux['AUX_sample_name'].eq('GJet')
    aux.loc[nonres_mask, 'AUX_eventWeightTrain'] = aux.loc[nonres_mask, 'AUX_eventWeightTrain'] * 1.78
    # Resonant background
    for i, _ in enumerate([key for key in filepaths.keys() if 'nonRes' not in key and 'HH' not in key]):
        class_mask = aux['AUX_label1D'].eq(i)
        aux.loc[class_mask, 'AUX_eventWeightTrain'] = aux.loc[class_mask, 'AUX_eventWeightTrain'] * RES_BKG_RESCALE
    # Signal
    for i, _ in enumerate([key for key in filepaths.keys() if 'HH' in key]):
        signal_mask = aux['AUX_label1D'].eq(i)
        background_mask = aux['AUX_label1D'].ne(i)
        aux.loc[signal_mask, 'AUX_eventWeightTrain'] = aux.loc[signal_mask, 'AUX_eventWeightTrain'] * np.sum(aux.loc[background_mask, 'AUX_eventWeightTrain']) / np.sum(aux.loc[signal_mask, 'AUX_eventWeightTrain'])
    
    if DF_SHUFFLE:
        rng = np.random.default_rng(seed=RNG_SEED)
        class_shuffle_idx = rng.permutation(df.index)
        df.reindex(class_shuffle_idx)
        aux.reindex(class_shuffle_idx)

    return df, aux

def get_DMatrix(df, aux, dataset: str='train'):
    return xgb.DMatrix(
        data=df, label=aux['AUX_label1D'], weight=np.abs(aux['AUX_eventWeightTrain'] if dataset.lower() == 'train' else aux['AUX_eventWeight']),
        missing=FILL_VALUE, feature_names=list(df.columns)
    )
def get_DMatrices(get_filepaths, fold_idx: int, val_split: float=0.2, **kwargs):
    if 'res_bkg_rescale' in kwargs: RES_BKG_RESCALE = kwargs['res_bkg_rescale']
    if 'shuffle' in kwargs: DF_SHUFFLE = kwargs['shuffle']
    tr_df, tr_aux = get_Dataframes(get_filepaths, fold_idx, 'train')
    train_df, val_df, train_aux, val_aux = train_test_split(tr_df, tr_aux, test_size=val_split, random_state=RNG_SEED)
    test_df, test_aux = get_Dataframes(get_filepaths, fold_idx, 'test')

    train_dm = get_DMatrix(train_df, train_aux)
    val_dm = get_DMatrix(val_df, val_aux)
    test_dm = get_DMatrix(test_df, test_aux, dataset='test')

    return train_dm, val_dm, test_dm
