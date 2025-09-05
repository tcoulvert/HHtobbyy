# Stdlib packages
import copy
import glob
import json
import re
import os

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

FILL_VALUE = -999

def process_data(
    filepaths_dict, output_dirpath, order,
    seed=21, mod_vals=(5, 5), k_fold_test=True, save=True,
    std_json_dirpath=None, other_bkg_rescale=5, jet_prefix='nonRes',
    apply_mask=True
):
    # Load parquet files #
    samples = {}
    for sample_name, sample_filepaths in filepaths_dict.items():
        # for dir_path in sample_filepaths:
        #     print(dir_path)
        #     print(glob.glob(dir_path))
        sample_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in sample_filepaths]
        if re.search('VH', sample_name) is not None:
            ZandWH_idxs = []
            for idx, dir_path in enumerate(sample_filepaths):
                if re.search('VH', dir_path) is None:
                    ZandWH_idxs.append(idx)
            for idx in ZandWH_idxs:
                if idx%2 == 0:
                    TIGHT_PNETBTAG_WP = 0.6734
                else:
                    TIGHT_PNETBTAG_WP = 0.6915
                sample_list[idx] = sample_list[idx][
                    (sample_list[idx][f'{jet_prefix}_lead_bjet_btagPNetB'] > TIGHT_PNETBTAG_WP)
                    & (sample_list[idx][f'{jet_prefix}_sublead_bjet_btagPNetB'] > TIGHT_PNETBTAG_WP)
                ]
        samples[sample_name] = ak.concatenate(sample_list)
        # for field in samples[sample_name].fields:
        #     print(field)
        #     print('-'*60)
        if apply_mask:
            samples[sample_name] = samples[sample_name][
                samples[sample_name][f'{jet_prefix}_has_two_btagged_jets']
                & (
                    samples[sample_name]['fiducialGeometricFlag'] if 'fiducialGeometricFlag' in samples[sample_name].fields else samples[sample_name]['pass_fiducial_geometric']
                ) & (
                    (
                        samples[sample_name]['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90']
                        & samples[sample_name]['Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95']
                    ) 
                    if 'Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90' in samples[sample_name].fields else (samples[sample_name]['mass'] > 0)
                ) & (
                    (samples[sample_name]['lead_mvaID'] > -0.7)
                    & (samples[sample_name]['sublead_mvaID'] > -0.7)
                )
            ]

    # Rescale factor for sig and bkg samples
    if len(filepaths_dict) > 1:
        sum_of_sig = np.sum(samples[order[0]]['eventWeight'])
        sum_of_bkg = np.sum(
            np.concatenate(
                [
                    samples[order[i]]['eventWeight']*(
                        other_bkg_rescale if (re.search('HH', order[i]) is None and re.search('non-res', order[i]) is None) else 1
                    ) for i in range(1, len(order))
                ], axis=0
            ), axis=None
        )
        sig_rescale_factor = (sum_of_bkg / sum_of_sig)
    else:
        sig_rescale_factor = -999
    
    # Convert parquet files to pandas DFs #
    pandas_samples = {}
    dont_include_vars = []
    merger_vars_map = {
        'nonRes': '',
        'nonResReg': 'MbbReg_',
        'nonResReg_DNNpair': 'MbbRegDNNPair_'
    }
    high_level_fields = {
        # MET variables
        'puppiMET_pt',

        # jet vars
        f'{jet_prefix}_DeltaPhi_j1MET', f'{jet_prefix}_DeltaPhi_j2MET', 
        'n_jets', f'{jet_prefix}_chi_t0', f'{jet_prefix}_chi_t1',

        # lepton vars
        'lepton1_pt', 'lepton1_pfIsoId', 'lepton1_mvaID',
        'DeltaR_b1l1',

        # angular vars
        f'{jet_prefix}_CosThetaStar_CS', f'{jet_prefix}_CosThetaStar_jj', f'{jet_prefix}_CosThetaStar_gg',

        # bjet vars
        f'{jet_prefix}_lead_bjet_eta', # eta
        # f'{jet_prefix}_lead_bjet_btagPNetB', # btag scores
        f"{jet_prefix}_lead_bjet_bTagWPL", f"{jet_prefix}_lead_bjet_bTagWPM", f"{jet_prefix}_lead_bjet_bTagWPT",
        f"{jet_prefix}_lead_bjet_bTagWPXT", f"{jet_prefix}_lead_bjet_bTagWPXXT",
        # --------
        f'{jet_prefix}_sublead_bjet_eta', 
        # f'{jet_prefix}_sublead_bjet_btagPNetB',
        f"{jet_prefix}_sublead_bjet_bTagWPL", f"{jet_prefix}_sublead_bjet_bTagWPM", f"{jet_prefix}_sublead_bjet_bTagWPT",
        f"{jet_prefix}_sublead_bjet_bTagWPXT", f"{jet_prefix}_sublead_bjet_bTagWPXXT",
        

        # diphoton vars
        'eta',

        # Photon vars
        'lead_mvaID', 'lead_sigmaE_over_E',
        # --------
        'sublead_mvaID', 'sublead_sigmaE_over_E',
        
        # HH vars
        f'{jet_prefix}_HHbbggCandidate_pt', 
        f'{jet_prefix}_HHbbggCandidate_eta', 
        f'{merger_vars_map[jet_prefix]}pt_balance',

        # ZH vars
        f'{merger_vars_map[jet_prefix]}DeltaPhi_jj',
        f'{merger_vars_map[jet_prefix]}DeltaPhi_isr_jet_z',
    }
    mHH_corr_high_level_fields = {
        # MET variables
        'puppiMET_sumEt',  #eft

        # jet vars
        f'{jet_prefix}_DeltaR_jg_min',  #eft

        # dijet vars
        f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),  #eft
        f'{jet_prefix}_dijet_pt',  #eft

        # bjet vars
        f'{jet_prefix}_lead_bjet_pt', #eft
        f'{merger_vars_map[jet_prefix]}lead_bjet_sigmapT_over_pT', #eft
        f'{merger_vars_map[jet_prefix]}lead_bjet_pt_over_Mjj', #eft
        # --------
        f'{jet_prefix}_sublead_bjet_pt', #eft
        f'{merger_vars_map[jet_prefix]}sublead_bjet_sigmapT_over_pT', #eft
        f'{merger_vars_map[jet_prefix]}sublead_bjet_pt_over_Mjj', #eft

        # diphoton vars
        'pt',  #eft

        # ZH vars
        f'{merger_vars_map[jet_prefix]}DeltaEta_jj', #eft
        f'{merger_vars_map[jet_prefix]}isr_jet_pt',  #eft
    }
    if re.search('_EFT', output_dirpath) is None: 
        high_level_fields = high_level_fields | mHH_corr_high_level_fields
    std_mapping = {
        # MET variables
        'puppiMET_sumEt': 'puppiMET_sumEt', 'puppiMET_pt': 'puppiMET_pt',

        # jet vars
        f'{jet_prefix}_DeltaPhi_j1MET': 'DeltaPhi_j1MET', f'{jet_prefix}_DeltaPhi_j2MET': 'DeltaPhi_j2MET', 
        f'{jet_prefix}_DeltaR_jg_min': 'DeltaR_jg_min', 
        'n_jets': 'n_jets', 
        f'{jet_prefix}_chi_t0': 'chi_t0', f'{jet_prefix}_chi_t1': 'chi_t1',

        # lepton vars
        'lepton1_pt': 'lepton1_pt', 'lepton1_pfIsoId': 'lepton1_pfIsoId', 'lepton1_mvaID': 'lepton1_mvaID',
        'DeltaR_b1l1': 'leadBjet_leadLepton',

        # angular vars
        f'{jet_prefix}_CosThetaStar_CS': 'CosThetaStar_CS', 
        f'{jet_prefix}_CosThetaStar_jj': 'CosThetaStar_jj', 
        f'{jet_prefix}_CosThetaStar_gg': 'CosThetaStar_gg',

        # dijet vars
        f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'): 'dijet_mass', 
        f'{jet_prefix}_dijet_pt': 'dijet_pt',

        # bjet vars
        f'{jet_prefix}_lead_bjet_pt': 'lead_bjet_pt', 
        f'{jet_prefix}_lead_bjet_eta': 'lead_bjet_eta',
        f'{jet_prefix}_lead_bjet_btagPNetB': 'lead_bjet_btagPNetB',
        f'{merger_vars_map[jet_prefix]}lead_bjet_sigmapT_over_pT': 'lead_bjet_sigmapT_over_pT', 
        f'{merger_vars_map[jet_prefix]}lead_bjet_pt_over_Mjj': 'lead_bjet_pt_over_Mjj',
        # --------
        f'{jet_prefix}_sublead_bjet_pt': 'sublead_bjet_pt',  
        f'{jet_prefix}_sublead_bjet_eta': 'sublead_bjet_eta',
        f'{jet_prefix}_sublead_bjet_btagPNetB': 'sublead_bjet_btagPNetB',
        f'{merger_vars_map[jet_prefix]}sublead_bjet_sigmapT_over_pT': 'sublead_bjet_sigmapT_over_pT', 
        f'{merger_vars_map[jet_prefix]}sublead_bjet_pt_over_Mjj': 'sublead_bjet_pt_over_Mjj',

        # diphoton vars
        'pt': 'pt', 'eta': 'eta',

        # Photon vars
        'lead_mvaID': 'lead_mvaID', 'lead_sigmaE_over_E': 'lead_sigmaE_over_E',
        # --------
        'sublead_mvaID': 'sublead_mvaID', 'sublead_sigmaE_over_E': 'sublead_sigmaE_over_E',
        
        # HH vars
        f'{jet_prefix}_HHbbggCandidate_pt': 'HHbbggCandidate_pt', 
        f'{jet_prefix}_HHbbggCandidate_eta': 'HHbbggCandidate_eta', 
        f'{merger_vars_map[jet_prefix]}pt_balance': 'pt_balance',

        # ZH vars
        f'{merger_vars_map[jet_prefix]}DeltaPhi_jj': 'DeltaPhi_jj', 
        f'{merger_vars_map[jet_prefix]}DeltaEta_jj': 'DeltaEta_jj',
        f'{merger_vars_map[jet_prefix]}DeltaPhi_isr_jet_z': 'DeltaPhi_isr_jet_z',
        f'{merger_vars_map[jet_prefix]}isr_jet_pt': 'isr_jet_pt',
        
        # Aux fields
        'event': 'event', 

        'mass': 'mass', f'{jet_prefix}_HHbbggCandidate_mass': 'HHbbggCandidate_mass',
        'lepton1_pt': 'lepton1_pt', 'lepton2_pt': 'lepton2_pt',

        'hash': 'hash', 'eventWeight': 'eventWeight', 'sample_name': 'sample_name', 

        f'{merger_vars_map[jet_prefix]}max_nonbjet_btag': 'max_nonbjet_btag',

        'MultiBDT_output': 'MultiBDT_output',
    }
    # for field in samples[order[0]].fields:
    #     print(field)
    #     print('-'*60)
        

    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'mass', f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),  # diphoton and bb-dijet mass
        f'{jet_prefix}_HHbbggCandidate_mass',
        'lepton1_pt', 'lepton2_pt',  # renamed to lepton1/2_bool in DataFrame, used to distinguish 0, 1, and 2+ lepton events
    }
    if 'hash' in samples[order[0]].fields:
        high_level_aux_fields.add('hash')  # for ensuring sorting of events after training/testing is performed
    if 'eventWeight' in samples[order[0]].fields:
        high_level_aux_fields.add('eventWeight')  # computed weight using (genWeight * lumi * xs / sum_of_genWeights)
    if 'sample_name' in samples[order[0]].fields:
        high_level_aux_fields.add('sample_name')
    if f'{merger_vars_map[jet_prefix]}max_nonbjet_btag' in samples[order[0]].fields:
        high_level_aux_fields.add(f'{merger_vars_map[jet_prefix]}max_nonbjet_btag')
    if 'MultiBDT_output' in samples[order[0]].fields:
        high_level_aux_fields.add('MultiBDT_output')


    hlf_list, hlf_aux_list = list(high_level_fields), list(high_level_aux_fields)
    hlf_list.sort()
    hlf_aux_list.sort()
    for sample_name, sample in samples.items():

        pandas_samples[sample_name] = {}
        for field in hlf_list:
            
            new_column = ak.to_numpy(sample[field], allow_missing=False).flatten()
            if new_column.dtype == np.float64:
                pandas_samples[sample_name][std_mapping[field]] = np.array(new_column, dtype=np.float32)
            else: 
                pandas_samples[sample_name][std_mapping[field]] = copy.deepcopy(new_column)
            
        pandas_aux_samples[sample_name] = {}
        for field in hlf_aux_list:

            new_column = ak.to_numpy(sample[field], allow_missing=False).flatten()
            if new_column.dtype == np.float64:
                pandas_aux_samples[sample_name][std_mapping[field]] = np.array(new_column, dtype=np.float32)
            else:
                pandas_aux_samples[sample_name][std_mapping[field]] = copy.deepcopy(new_column)
                
        # Compute bool for easy lepton-veto checks
        for old_field, new_field in [('lepton1_pt', 'lepton1_bool'), ('lepton2_pt', 'lepton2_bool')]:
            pandas_aux_samples[sample_name][new_field] = copy.deepcopy(pandas_aux_samples[sample_name][old_field] != FILL_VALUE)
            del pandas_aux_samples[sample_name][old_field]
        if 'MultiBDT_output' in high_level_aux_fields:
            pandas_aux_samples[sample_name]['MultiBDT_output_ggH'] = pandas_aux_samples[sample_name]['MultiBDT_output'][0::4]
            pandas_aux_samples[sample_name]['MultiBDT_output_ttH'] = pandas_aux_samples[sample_name]['MultiBDT_output'][1::4]
            pandas_aux_samples[sample_name]['MultiBDT_output_VH'] = pandas_aux_samples[sample_name]['MultiBDT_output'][2::4]
            pandas_aux_samples[sample_name]['MultiBDT_output_nonRes'] = pandas_aux_samples[sample_name]['MultiBDT_output'][3::4]
            pandas_aux_samples[sample_name].pop('MultiBDT_output')
        # for var, col in pandas_samples[sample_name].items():
        #     print('-'*60)
        #     print(var)
        #     print(np.shape(col))
        # for var, col in pandas_aux_samples[sample_name].items():
        #     print('-'*60)
        #     print(var)
        #     print(np.shape(col))
        pandas_samples[sample_name] = pd.DataFrame(pandas_samples[sample_name])
        pandas_aux_samples[sample_name] = pd.DataFrame(pandas_aux_samples[sample_name])

    if len(dont_include_vars) > 0:
        for var in dont_include_vars:
            if var not in high_level_fields:
                continue
            high_level_fields.remove(var)
        hlf_list = list(high_level_fields)
        hlf_list.sort()

    # Randomly shuffle DFs and split into train and test samples #
    rng = np.random.default_rng(seed=seed)
    for sample_name in pandas_samples.keys():
        idx = rng.permutation(pandas_samples[sample_name].index)
        pandas_samples[sample_name] = pandas_samples[sample_name].reindex(idx)
        pandas_aux_samples[sample_name] = pandas_aux_samples[sample_name].reindex(idx)

    def train_test_split_df(dict_of_dfs, dict_of_aux_dfs, dataset_num=0):
        # Train/Val events are those with eventID % mod_val != fold, test events are the others
        train_dict_of_dfs, test_dict_of_dfs, train_dict_of_aux_dfs, test_dict_of_aux_dfs = {}, {}, {}, {}
        for sample_name in dict_of_dfs.keys():
            train_mask = (dict_of_aux_dfs[sample_name]['event'] % mod_vals[0]).ne(dataset_num)
            test_mask = (dict_of_aux_dfs[sample_name]['event'] % mod_vals[0]).eq(dataset_num)

            train_dict_of_dfs[sample_name] = dict_of_dfs[sample_name].loc[train_mask].reset_index(drop=True)
            test_dict_of_dfs[sample_name] = dict_of_dfs[sample_name].loc[test_mask].reset_index(drop=True)
            train_dict_of_aux_dfs[sample_name] = dict_of_aux_dfs[sample_name].loc[train_mask].reset_index(drop=True)
            test_dict_of_aux_dfs[sample_name] = dict_of_aux_dfs[sample_name].loc[test_mask].reset_index(drop=True)

        return train_dict_of_dfs, test_dict_of_dfs, train_dict_of_aux_dfs, test_dict_of_aux_dfs
    
    for fold in range(mod_vals[0] if k_fold_test else 1):

        (
            train_dict_of_dfs, test_dict_of_dfs, 
            train_dict_of_aux_dfs, test_dict_of_aux_dfs
        ) = train_test_split_df(pandas_samples, pandas_aux_samples, dataset_num=fold)

        if len(dont_include_vars) > 0:
            
            for sample_name in train_dict_of_dfs.keys():
                if re.search('two_lepton_veto', output_dirpath) is not None:
                    train_slice = ~train_dict_of_dfs[sample_name]['lepton2_bool']
                    test_slice = ~test_dict_of_dfs[sample_name]['lepton2_bool']
                elif re.search('one_lepton_veto', output_dirpath) is not None:
                    train_slice = ~train_dict_of_dfs[sample_name]['lepton1_bool']
                    test_slice = ~test_dict_of_dfs[sample_name]['lepton1_bool']
                else:
                    train_slice = (train_dict_of_dfs[sample_name]['pt'] >= -999)
                    test_slice = (test_dict_of_dfs[sample_name]['pt'] >= -999)

                train_dict_of_dfs[sample_name] = train_dict_of_dfs[sample_name].loc[train_slice, hlf_list].reset_index(drop=True)
                train_dict_of_aux_dfs[sample_name] = train_dict_of_aux_dfs[sample_name].loc[train_slice].reset_index(drop=True)

                test_dict_of_dfs[sample_name] = test_dict_of_dfs[sample_name].loc[test_slice, hlf_list].reset_index(drop=True)
                test_dict_of_aux_dfs[sample_name] = test_dict_of_aux_dfs[sample_name].loc[test_slice].reset_index(drop=True)


        # Perform the standardization #
        no_standardize = {
            'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',
            'lead_bjet_eta', 'lead_bjet_phi',
            'sublead_bjet_eta', 'sublead_bjet_phi',
            'n_leptons',
            # Yibo BDT variables #
            'lead_mvaID', 'sublead_mvaID',
            'CosThetaStar_gg',
            # 'lead_bjet_btagPNetB', 'sublead_bjet_btagPNetB',
            # Michael's DNN variables #
            'HHbbggCandidate_eta', 'HHbbggCandidate_phi',
            # VH variables #
            'DeltaPhi_jj', 'DeltaPhi_isr_jet_z', 'DeltaEta_jj',
        }
        log_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', # MET variables
            # 'chi_t0', 
            'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lead_bjet_pt', 'sublead_bjet_pt', # bjet pts
            'HHbbggCandidate_pt', 'HHbbggCandidate_mass',  # HH object fields
            # VH and ATLAS variables #
            'dijet_pt', 'isr_jet_pt', 'pt_balance'
        }
        exp_fields = {
            ''
        }
        def apply_log_and_exp(df):
            for field in log_fields & high_level_fields:
                mask = (df.loc[:, field].to_numpy() > 0)
                df.loc[mask, field] = np.log(df.loc[mask, field])

            for field in exp_fields & high_level_fields:
                mask = (df.loc[:, field].to_numpy() != FILL_VALUE)
                df.loc[mask, field] = np.exp(df.loc[mask, field])

            return df
        

        # Because of zero-padding, standardization needs special treatment
        df_train = pd.concat([train_dict_of_dfs[sample_name] for sample_name in order], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_train = apply_log_and_exp(df_train)
        masked_x_sample = np.ma.array(df_train, mask=(df_train == FILL_VALUE))

        if std_json_dirpath is not None:
            std_json_filepath = glob.glob(std_json_dirpath+f'/*{fold}_standardization.json')[0]
            with open(std_json_filepath, 'r') as f:
                standardized_to_json = json.load(f)

            assert np.all(standardized_to_json['standardized_variables'] == df_train.columns), f"columns don't match -> std - DF cols \n{set(standardized_to_json['standardized_mean']) - set(df_train.columns)} \nand DF - std cols \nstd - DF cols \n{set(df_train.columns) - set(standardized_to_json['standardized_mean'])}"
            x_mean = standardized_to_json['standardized_mean']
            x_std = standardized_to_json['standardized_stddev']
        
        else:
            x_mean = masked_x_sample.mean(axis=0)
            x_std = masked_x_sample.std(axis=0)
            for i, col in enumerate(df_train.columns):
                if col in no_standardize:
                    x_mean[i] = 0
                    x_std[i] = 1


            standardized_to_json = {
                'standardized_logs': [True if col in log_fields else False for col in df_train.columns],
                'standardized_exps': [True if col in exp_fields else False for col in df_train.columns],
                'standardized_variables': [col for col in df_train.columns],
                'standardized_mean': [float(mean) for mean in x_mean],
                'standardized_stddev': [float(std) for std in x_std]
            }
            if save:
                with open(os.path.join(output_dirpath, f'MultiBDT_{fold}_standardization.json'), 'w') as f:
                    json.dump(standardized_to_json, f)

        # Standardize samples
        std_train_dict_of_dfs, std_test_dict_of_dfs = {}, {}
        for sample_name in order:
            std_train_df = apply_log_and_exp(copy.deepcopy(train_dict_of_dfs[sample_name]))
            normed_train = (np.ma.array(std_train_df, mask=(std_train_df == FILL_VALUE)) - x_mean)/x_std
            std_train_dict_of_dfs[sample_name] = pd.DataFrame(normed_train.filled(FILL_VALUE), columns=list(df_train))

            std_test_df = apply_log_and_exp(copy.deepcopy(test_dict_of_dfs[sample_name]))
            normed_test = (np.ma.array(std_test_df, mask=(std_test_df == FILL_VALUE)) - x_mean)/x_std
            std_test_dict_of_dfs[sample_name] = pd.DataFrame(normed_test.filled(FILL_VALUE), columns=list(df_train))

        column_list = [col_name for col_name in df_train.columns]
        hlf_vars_columns = {col_name: i for i, col_name in enumerate(column_list)}


        # Build pre-std DF
        train_df = pd.concat([train_dict_of_dfs[sample_name] for sample_name in order], ignore_index=True)
        train_aux_df = pd.concat([train_dict_of_aux_dfs[sample_name] for sample_name in order], ignore_index=True)
        test_df = pd.concat([test_dict_of_dfs[sample_name] for sample_name in order], ignore_index=True)
        test_aux_df = pd.concat([test_dict_of_aux_dfs[sample_name] for sample_name in order], ignore_index=True)

    
        def get_labels():
            if len(filepaths_dict) > 2:
                train_labels, test_labels = [], []
                for i, sample_name in enumerate(order):
                    sample_label = [0 if j != i else 1 for j in range(len(order))]
                    train_labels.append(np.tile(sample_label, (np.shape(std_train_dict_of_dfs[sample_name])[0], 1)))
                    test_labels.append(np.tile(sample_label, (np.shape(std_test_dict_of_dfs[sample_name])[0], 1)))
            else:
                train_labels, test_labels = [], []
                for i, sample_name in enumerate(order):
                    train_labels.append(np.ones(np.shape(std_train_data)[0]) if i == 0 else np.zeros(np.shape(std_train_data)[0]))
                    test_labels.append(np.ones(np.shape(std_test_data)[0]) if i == 0 else np.zeros(np.shape(std_test_data)[0]))

            return np.concatenate(train_labels), np.concatenate(test_labels)

        # Build data arrays
        std_train_data = pd.concat([std_train_dict_of_dfs[sample_name] for sample_name in order], ignore_index=True).values
        std_test_data = pd.concat([std_test_dict_of_dfs[sample_name] for sample_name in order], ignore_index=True).values
        # Build labels
        train_labels, test_labels = get_labels()

        # Shuffle train arrays
        p = rng.permutation(len(std_train_data))
        std_train_data, train_labels = std_train_data[p], train_labels[p]
        # Shuffle train DFs
        train_df = (train_df.reindex(p)).reset_index(drop=True)
        train_aux_df = (train_aux_df.reindex(p)).reset_index(drop=True)
        # print("Data HLF: {}".format(std_train_data.shape))
        # for sample_name in order:
        #     print(f"num {sample_name} = {np.shape(std_train_dict_of_dfs[sample_name].values)[0]}")
        # print(f"n signal = {len(label[label == 1])}, n bkg = {len(label[label == 0])}")

        # Shuffle test arrays
        p_test = rng.permutation(len(std_test_data))
        std_test_data, test_labels = std_test_data[p_test], test_labels[p_test]
        # Build and shuffle test DFs
        test_df = (test_df.reindex(p_test)).reset_index(drop=True)
        test_aux_df = (test_aux_df.reindex(p_test)).reset_index(drop=True)
        # print("Data HLF test: {}".format(std_test_data.shape))
        # for sample_name in order:
        #     print(f"num {sample_name} = {np.shape(std_test_dict_of_dfs[sample_name].values)[0]}")
        # print(f"n signal = {len(label_test[label_test == 1])}, n bkg = {len(label_test[label_test == 0])}")

        if not k_fold_test:
            return (
                sig_rescale_factor,
                train_df, test_df, 
                std_train_data, train_labels, 
                std_test_data, test_labels, 
                hlf_vars_columns,
                train_aux_df, test_aux_df
            )
        elif k_fold_test and fold == 0:
            (
                full_data_df, full_data_test_df, 
                full_data_hlf, full_label, 
                full_data_hlf_test, full_label_test, 
                full_hlf_vars_columns,
                full_data_aux, full_data_test_aux
            ) = (
                {f'fold_{0}': copy.deepcopy(train_df)}, {f'fold_{0}': copy.deepcopy(test_df)}, 
                {f'fold_{0}': copy.deepcopy(std_train_data)}, {f'fold_{0}': copy.deepcopy(train_labels)}, 
                {f'fold_{0}': copy.deepcopy(std_test_data)}, {f'fold_{0}': copy.deepcopy(test_labels)}, 
                {f'fold_{0}': copy.deepcopy(hlf_vars_columns)},
                {f'fold_{0}': copy.deepcopy(train_aux_df)}, {f'fold_{0}': copy.deepcopy(test_aux_df)}
            )
        else:
            (
                full_data_df[f'fold_{fold}'], full_data_test_df[f'fold_{fold}'], 
                full_data_hlf[f'fold_{fold}'], full_label[f'fold_{fold}'], 
                full_data_hlf_test[f'fold_{fold}'], full_label_test[f'fold_{fold}'], 
                full_hlf_vars_columns[f'fold_{fold}'],
                full_data_aux[f'fold_{fold}'], full_data_test_aux[f'fold_{fold}']
            ) = (
                copy.deepcopy(train_df), copy.deepcopy(test_df), 
                copy.deepcopy(std_train_data), copy.deepcopy(train_labels), 
                copy.deepcopy(std_test_data), copy.deepcopy(test_labels), 
                copy.deepcopy(hlf_vars_columns),
                copy.deepcopy(train_aux_df), copy.deepcopy(test_aux_df)
            )

            if k_fold_test and fold == (mod_vals[0] - 1):
                return (
                    sig_rescale_factor,
                    full_data_df, full_data_test_df, 
                    full_data_hlf, full_label, 
                    full_data_hlf_test, full_label_test, 
                    full_hlf_vars_columns,
                    full_data_aux, full_data_test_aux
                )

            


