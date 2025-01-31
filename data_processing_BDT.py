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
    std_json_dirpath=None, other_bkg_rescale=5
):
    # Load parquet files #
    samples = {}
    for sample_name, sample_filepaths in filepaths_dict.items():
        sample_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in sample_filepaths]
        samples[sample_name] = ak.concatenate(sample_list)

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
        sig_rescale_factor = sum_of_bkg / sum_of_sig
    else:
        sig_rescale_factor = -999
    
    # Convert parquet files to pandas DFs #
    pandas_samples = {}
    dont_include_vars = []
    high_level_fields = {
        'puppiMET_sumEt', 'puppiMET_pt', # MET variables
        'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
        'DeltaR_jg_min', 'n_jets', 
        'chi_t0', 'chi_t1', # jet variables
        'lepton1_pt', 'lepton1_eta', 
        'lepton2_pt', 'lepton2_eta',  # lepton pt, eta
        'pt', 'eta',  # diphoton pt, eta
        'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
        'dijet_mass', # mass of b-dijet (resonance for H->bb)
        'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
        'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
        'n_leptons', 
        'lead_bjet_pt', 'lead_bjet_eta',
        'sublead_bjet_pt', 'sublead_bjet_eta',
        # Yibos BDT variables #
        'lead_mvaID', 'sublead_mvaID',
        'CosThetaStar_gg',
        'lead_sigmaE_over_E', 'sublead_sigmaE_over_E',
        'lead_bjet_pt_over_Mjj', 'sublead_bjet_pt_over_Mjj',
        'lead_bjet_btagPNetB', 'sublead_bjet_btagPNetB',
        'lead_bjet_sigmapT_over_pT', 'sublead_bjet_sigmapT_over_pT',
        'dijet_mass_over_Mggjj',
        # Michael's DNN variables #
        'HHbbggCandidate_pt', 'HHbbggCandidate_eta',
        # ATLAS variables #
        'pt_balance',
        # ZH variables #
        'DeltaPhi_jj', 'DeltaEta_jj',
        'isr_jet_pt', 'DeltaPhi_isr_jet_z',
        'dijet_pt',
    }
    if re.search('two_lepton_veto', output_dirpath) is not None:
        dont_include_vars.extend([
            'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
            'leadBjet_subleadLepton', 'subleadBjet_subleadLepton',
        ])
    elif re.search('one_lepton_veto', output_dirpath) is not None:
        dont_include_vars.extend([
            'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
            'leadBjet_leadLepton', 'subleadBjet_leadLepton',
            'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
            'leadBjet_subleadLepton', 'subleadBjet_subleadLepton',
        ])
    if re.search('no_photonIso', output_dirpath) is not None:
        dont_include_vars.extend([
            'DeltaR_j1g1', 'DeltaR_j1g2', 'DeltaR_j2g1', 'DeltaR_j2g2', 'DeltaR_jg_min',
            'lead_pfRelIso03_all_quadratic', 'sublead_pfRelIso03_all_quadratic',
        ])
    if re.search('no_HH', output_dirpath) is not None:
        dont_include_vars.extend([
            'HHbbggCandidate_pt', 'HHbbggCandidate_eta', 'HHbbggCandidate_phi',
            'HHbbggCandidate_mass'
        ])
    if re.search('no_leptonIso', output_dirpath) is not None:
        dont_include_vars.extend([
            'leadBjet_leadLepton', 'leadBjet_subleadLepton',
            'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
        ])
    if re.search('no_diphoton', output_dirpath) is not None:
        dont_include_vars.extend([
            'pt', 'eta', 'phi', 'dipho_mass_over_Mggjj'
        ])
    if re.search('no_diphoMass', output_dirpath) is not None:
        dont_include_vars.extend([
            'dipho_mass_over_Mggjj', 'lead_pt_over_Mgg', 'sublead_pt_over_Mgg',
        ])
    if re.search('no_diphoPt', output_dirpath) is not None:
        dont_include_vars.extend([
            'pt'
        ])
    if re.search('no_diphoEta', output_dirpath) is not None:
        dont_include_vars.extend([
            'eta'
        ])
    if re.search('no_diphoPhi', output_dirpath) is not None:
        dont_include_vars.extend([
            'phi'
        ])
    if re.search('no_lepton', output_dirpath) is not None:
        dont_include_vars.extend([
            'lepton1_pt', 'lepton2_pt', 'lepton1_eta', 'lepton2_eta',
            'lepton1_phi', 'lepton2_phi', 'n_leptons',
        ])
    if re.search('no_MET', output_dirpath) is not None:
        dont_include_vars.extend([
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_phi',
        ])
    if re.search('no_badVars', output_dirpath) is not None:
        dont_include_vars.extend([
            'lepton2_pt', 'lepton2_eta',
            'leadBjet_subleadLepton', 'subleadBjet_subleadLepton',
            'lepton1_eta', 'n_leptons', 'subleadBjet_leadLepton',
            'dijet_mass_over_Mggjj',
            # 'chi_t1', 'leadBjet_leadLepton',
        ])
    if re.search('no_badMyyVars', output_dirpath) is not None:
        dont_include_vars.extend([
            'dijet_mass_over_Mggjj', 'puppiMET_sumEt'
        ])
    
    if (
        re.search('v2', output_dirpath) is not None
        and re.search('nonres_and_ttH_and_DNN_vars$', output_dirpath[:output_dirpath.rfind('/')]) is not None
    ):
        high_level_fields.add('puppiMET_eta')
        high_level_fields.add('HHbbggCandidate_mass')
    elif (
        re.search('v3', output_dirpath) is not None
    ):
        high_level_fields.add('HHbbggCandidate_phi')
    elif (
        re.search('v4', output_dirpath) is not None
        and re.search('no_diphoPhi', output_dirpath) is None
    ):
        high_level_fields.add('phi')
    elif (
        re.search('v5', output_dirpath) is not None
        and re.search('no_photonIso', output_dirpath) is not None
    ):
        high_level_fields.add('DeltaR_jg_min')
    elif (
        re.search('v1', output_dirpath) is not None
        or re.search('v2', output_dirpath) is not None
        or re.search('v3', output_dirpath) is not None
        or re.search('v4', output_dirpath) is not None
    ):
        if re.search('no_diphoMass', output_dirpath) is None:
            high_level_fields.add('dipho_mass_over_Mggjj')
        if re.search('no_photonIso', output_dirpath) is None:
            high_level_fields.add('DeltaR_j1g1')
            high_level_fields.add('DeltaR_j1g2')
            high_level_fields.add('DeltaR_j2g1')
            high_level_fields.add('DeltaR_j2g2')
        # photons pt / Myy
        high_level_fields.add('lead_pt_over_Mgg')
        high_level_fields.add('sublead_pt_over_Mgg')
        # lepton phi
        high_level_fields.add('lepton1_phi')
        high_level_fields.add('lepton2_phi')
        # bjets phi
        high_level_fields.add('lead_bjet_phi')
        high_level_fields.add('sublead_bjet_phi')
        # puppiMET phi
        high_level_fields.add('puppiMET_phi')
        

    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'mass', 'dijet_mass',  # diphoton and bb-dijet mass
        'HHbbggCandidate_mass',
        'lepton1_pt', 'lepton2_pt',  # renamed to lepton1/2_bool in DataFrame, used to distinguish 0, 1, and 2+ lepton events
    }
    if 'hash' in samples[order[0]].fields:
        high_level_aux_fields.add('hash')  # for ensuring sorting of events after training/testing is performed
    if 'eventWeight' in samples[order[0]].fields:
        high_level_aux_fields.add('eventWeight')  # computed eventWeight using (genWeight * lumi * xs / sum_of_genWeights)
    if 'sample_name' in samples[order[0]].fields:
        high_level_aux_fields.add('sample_name')  # computed eventWeight using (genWeight * lumi * xs / sum_of_genWeights)

    hlf_list, hlf_aux_list = list(high_level_fields), list(high_level_aux_fields)
    hlf_list.sort()
    hlf_aux_list.sort()
    for sample_name, sample in samples.items():
        pandas_samples[sample_name] = pd.DataFrame({
            field: ak.to_numpy(sample[field], allow_missing=False) for field in hlf_list
        })
        pandas_aux_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in hlf_aux_list
        }
        # Compute bool for easy lepton-veto checks
        for old_field, new_field in [('lepton1_pt', 'lepton1_bool'), ('lepton2_pt', 'lepton2_bool')]:
            pandas_aux_samples[sample_name][new_field] = copy.deepcopy(pandas_aux_samples[sample_name][old_field] != FILL_VALUE)
            del pandas_aux_samples[sample_name][old_field]
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
                    train_slice = (train_dict_of_dfs[sample_name]['lepton2_pt'] == -999)
                    test_slice = (test_dict_of_dfs[sample_name]['lepton2_pt'] == -999)
                elif re.search('one_lepton_veto', output_dirpath) is not None:
                    train_slice = (train_dict_of_dfs[sample_name]['lepton1_pt'] == -999)
                    test_slice = (test_dict_of_dfs[sample_name]['lepton1_pt'] == -999)
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
            'DeltaPhi_jj', 'DeltaPhi_isr_jet_z',
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

            


