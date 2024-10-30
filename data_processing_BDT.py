# Stdlib packages
import copy
import glob
import json
import re

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

import matplotlib.pyplot as plt

FILL_VALUE = -999

def process_data(
    signal_filepaths, bkg_filepaths, output_dirpath, 
    seed=None, mod_vals=(2, 2), k_fold_test=False
):
    # Load parquet files #
    
    sig_samples_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in signal_filepaths]
    sig_samples_pq = ak.concatenate(sig_samples_list)
    bkg_samples_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in bkg_filepaths]
    bkg_samples_pq = ak.concatenate(bkg_samples_list)
    samples = {
        'sig': sig_samples_pq,
        'bkg': bkg_samples_pq,
    }
    # for sample in samples.values():
    #     sample['n_leptons'] = ak.where(sample['n_leptons'] == -999, 0, sample['n_leptons'])
    
    # Convert parquet files to pandas DFs #
    pandas_samples = {}
    dont_include_vars = []
    high_level_fields = {
        'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
        'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
        'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
        'lepton1_pt', 'lepton2_pt', 'pt', # lepton and diphoton pt
        'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
        'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
        'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
        'dijet_mass', # mass of b-dijet (resonance for H->bb)
        'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
        'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
        'n_leptons', 
        'lead_bjet_pt', 'lead_bjet_eta', 'lead_bjet_phi',
        'sublead_bjet_pt', 'sublead_bjet_eta', 'sublead_bjet_phi',
        # Yibos BDT variables #
        'lead_mvaID', 'sublead_mvaID',
        'CosThetaStar_gg',
        'lead_pt_over_Mgg', 'sublead_pt_over_Mgg',
        'lead_sigmaE_over_E', 'sublead_sigmaE_over_E',
        'lead_bjet_pt_over_Mgg', 'sublead_bjet_pt_over_Mgg',
        'lead_bjet_btagPNetB', 'sublead_bjet_btagPNetB',
        'lead_bjet_sigmapT_over_pT', 'sublead_bjet_sigmapT_over_pT',
        'dipho_mass_over_Mggjj', 'dijet_mass_over_Mggjj',
        # My variables for non-reso reduction #
        'lead_pfRelIso03_all_quadratic', 'sublead_pfRelIso03_all_quadratic',
    }
    if re.search('two_lepton_veto', output_dirpath) is not None:
        dont_include_vars = [
            'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
            'leadBjet_subleadLepton', 'subleadBjet_subleadLepton',
        ]
    elif re.search('one_lepton_veto', output_dirpath) is not None:
        dont_include_vars = [
            'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
            'leadBjet_leadLepton', 'subleadBjet_leadLepton',
            'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
            'leadBjet_subleadLepton', 'subleadBjet_subleadLepton',
        ]

    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'eventWeight',  # computed eventWeight using (genWeight * lumi * xs / sum_of_genWeights)
        'mass', 'dijet_mass',  # diphoton and bb-dijet mass
        'lepton1_pt', 'lepton2_pt',  # renamed to lepton1/2_bool in DataFrame, used to distinguish 0, 1, and 2+ lepton events
    } # https://stackoverflow.com/questions/67003141/how-to-remove-a-field-from-a-collection-of-records-created-by-awkward-zip

    hlf_list, hlf_aux_list = list(high_level_fields), list(high_level_aux_fields)
    hlf_list.sort()
    hlf_aux_list.sort()
    for sample_name, sample in samples.items():
        pandas_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in hlf_list
        }
        pandas_aux_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in hlf_aux_list
        }
        # Compute bool for easy lepton-veto checks
        for old_field, new_field in [('lepton1_pt', 'lepton1_bool'), ('lepton2_pt', 'lepton2_bool')]:
            pandas_aux_samples[sample_name][new_field] = copy.deepcopy(pandas_aux_samples[sample_name][old_field] != FILL_VALUE)
            del pandas_aux_samples[sample_name][old_field]

    sig_frame = pd.DataFrame(pandas_samples['sig'])
    sig_aux_frame = pd.DataFrame(pandas_aux_samples['sig'])
    bkg_frame = pd.DataFrame(pandas_samples['bkg'])
    bkg_aux_frame = pd.DataFrame(pandas_aux_samples['bkg'])

    # Randomly shuffle DFs and split into train and test samples #
    rng = np.random.default_rng(seed=seed)
    sig_idx = rng.permutation(sig_frame.index)
    bkg_idx = rng.permutation(bkg_frame.index)
    sig_frame = sig_frame.reindex(sig_idx)
    sig_aux_frame = sig_aux_frame.reindex(sig_idx)
    bkg_frame = bkg_frame.reindex(bkg_idx)
    bkg_aux_frame = bkg_aux_frame.reindex(bkg_idx)

    def train_test_split_df(sig_df, sig_aux_df, bkg_df, bkg_aux_df, method='modulus', dataset_num=0):
        if method == 'modulus':
            # Train/Val events are those with odd event #s, test events have even event #s
            sig_train_frame = sig_df.loc[(sig_aux_df['event'] % mod_vals[0]).ne(dataset_num)].reset_index(drop=True)
            sig_test_frame = sig_df.loc[(sig_aux_df['event'] % mod_vals[1]).eq(dataset_num)].reset_index(drop=True)

            sig_aux_train_frame = sig_aux_df.loc[(sig_aux_df['event'] % mod_vals[0]).ne(dataset_num)].reset_index(drop=True)
            sig_aux_test_frame = sig_aux_df.loc[(sig_aux_df['event'] % mod_vals[1]).eq(dataset_num)].reset_index(drop=True)

            bkg_train_frame = bkg_df.loc[(bkg_aux_df['event'] % mod_vals[0]).ne(dataset_num)].reset_index(drop=True)
            bkg_test_frame = bkg_df.loc[(bkg_aux_df['event'] % mod_vals[1]).eq(dataset_num)].reset_index(drop=True)

            bkg_aux_train_frame = bkg_aux_df.loc[(bkg_aux_df['event'] % mod_vals[0]).ne(dataset_num)].reset_index(drop=True)
            bkg_aux_test_frame = bkg_aux_df.loc[(bkg_aux_df['event'] % mod_vals[1]).eq(dataset_num)].reset_index(drop=True)
        else:
            raise Exception(f"Only 2 accepted methods: 'sample' and 'modulus'. You input {method}")
        return sig_train_frame, sig_test_frame, sig_aux_train_frame, sig_aux_test_frame, bkg_train_frame, bkg_test_frame, bkg_aux_train_frame, bkg_aux_test_frame
    
    for fold in range(mod_vals[0] if k_fold_test else 1):

        (
            sig_train_frame, sig_test_frame, 
            sig_aux_train_frame, sig_aux_test_frame, 
            bkg_train_frame, bkg_test_frame,
            bkg_aux_train_frame, bkg_aux_test_frame
        ) = train_test_split_df(sig_frame, sig_aux_frame, bkg_frame, bkg_aux_frame, dataset_num=fold)

        ## Further selection for lepton-veto check ##
        if len(dont_include_vars) > 0:
            keep_cols = list(high_level_fields - set(dont_include_vars))

            if re.search('two_lepton_veto', output_dirpath) is not None:
                sig_train_slice = (sig_train_frame['lepton2_pt'] == -999)
                bkg_train_slice = (bkg_train_frame['lepton2_pt'] == -999)
                # sig_test_slice = (sig_test_frame['lepton2_pt'] == -999)
                # bkg_test_slice = (bkg_test_frame['lepton2_pt'] == -999)
            elif re.search('one_lepton_veto', output_dirpath) is not None:
                sig_train_slice = (sig_train_frame['lepton1_pt'] == -999)
                bkg_train_slice = (bkg_train_frame['lepton1_pt'] == -999)
                # sig_test_slice = (sig_test_frame['lepton1_pt'] == -999)
                # bkg_test_slice = (bkg_test_frame['lepton1_pt'] == -999)
            
            sig_train_frame = sig_train_frame.loc[sig_train_slice, keep_cols].reset_index(drop=True)
            bkg_train_frame = bkg_train_frame.loc[bkg_train_slice, keep_cols].reset_index(drop=True)
            sig_aux_train_frame = sig_aux_train_frame.loc[sig_train_slice].reset_index(drop=True)
            bkg_aux_train_frame = bkg_aux_train_frame.loc[bkg_train_slice].reset_index(drop=True)

            sig_test_frame = sig_test_frame[keep_cols].reset_index(drop=True)
            bkg_test_frame = bkg_test_frame[keep_cols].reset_index(drop=True)
            # sig_test_frame = sig_test_frame[sig_test_slice, keep_cols].reset_index(drop=True)
            # bkg_test_frame = bkg_test_frame[bkg_test_slice, keep_cols].reset_index(drop=True)
            # sig_aux_test_frame = sig_aux_test_frame.loc[sig_test_slice].reset_index(drop=True)
            # bkg_aux_test_frame = bkg_aux_test_frame.loc[bkg_test_slice].reset_index(drop=True)

            for var in dont_include_vars:
                high_level_fields.remove(var)
        ## End further selection for lepton-veto check ##


        # Perform the standardization #
        no_standardize = {
            'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',
            'lead_bjet_eta', 'lead_bjet_phi',
            'sublead_bjet_eta', 'sublead_bjet_phi',
            # Yibo BDT variables #
            'lead_mvaID', 'sublead_mvaID',
            'CosThetaStar_gg',
            'lead_bjet_btagPNetB', 'sublead_bjet_btagPNetB',
        }
        log_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', # MET variables
            'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lead_bjet_pt', 'sublead_bjet_pt', # bjet pts
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
        }
        exp_fields = {
            'lead_sigmaE_over_E', 'sublead_sigmaE_over_E',
            'lead_bjet_sigmapT_over_pT', 'sublead_bjet_sigmapT_over_pT',
        }
        def apply_log_and_exp(df):
            for field in log_fields & high_level_fields:
                mask = (df.loc[:, field].to_numpy() > 0)
                df.loc[mask, field] = np.log(df[mask, field])

            for field in exp_fields & high_level_fields:
                mask = (df.loc[:, field].to_numpy() != FILL_VALUE)
                df.loc[mask, field] = np.exp(df.loc[mask, field])

            return df
        
        # Because of zero-padding, standardization needs special treatment
        df_train = pd.concat([sig_train_frame, bkg_train_frame], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_train = apply_log_and_exp(copy.deepcopy(df_train))
        masked_x_sample = np.ma.array(df_train, mask=(df_train == FILL_VALUE))
        x_mean = masked_x_sample.mean(axis=0)
        x_std = masked_x_sample.std(axis=0)
        for i, col in enumerate(df_train.columns):
            if col in no_standardize:
                x_mean[i] = 0
                x_std[i] = 1

        # Standardize background
        normed_bkg_train_frame = apply_log_and_exp(copy.deepcopy(bkg_train_frame))
        normed_bkg_train = (np.ma.array(normed_bkg_train_frame, mask=(normed_bkg_train_frame == FILL_VALUE)) - x_mean)/x_std
        normed_bkg_test_frame = apply_log_and_exp(copy.deepcopy(bkg_test_frame))
        normed_bkg_test = (np.ma.array(normed_bkg_test_frame, mask=(normed_bkg_test_frame == FILL_VALUE)) - x_mean)/x_std
        normed_bkg_train_frame = pd.DataFrame(normed_bkg_train.filled(FILL_VALUE), columns=list(bkg_train_frame))
        normed_bkg_test_frame = pd.DataFrame(normed_bkg_test.filled(FILL_VALUE), columns=list(bkg_test_frame))

        # Standardize signal
        normed_sig_train_frame = apply_log_and_exp(copy.deepcopy(sig_train_frame))
        normed_sig_train = (np.ma.array(normed_sig_train_frame, mask=(normed_sig_train_frame == FILL_VALUE)) - x_mean)/x_std
        normed_sig_test_frame = apply_log_and_exp(copy.deepcopy(sig_test_frame))
        normed_sig_test = (np.ma.array(normed_sig_test_frame, mask=(normed_sig_test_frame == FILL_VALUE)) - x_mean)/x_std
        normed_sig_train_frame = pd.DataFrame(normed_sig_train.filled(FILL_VALUE), columns=list(sig_train_frame))
        normed_sig_test_frame = pd.DataFrame(normed_sig_test.filled(FILL_VALUE), columns=list(sig_test_frame))

        standardized_to_json = {
            'standardized_variables': [col for col in df_train.columns],
            'standardized_mean': [float(mean) for mean in x_mean],
            'standardized_stddev': [float(std) for std in x_std],
            'standardized_unphysical_values': [float(FILL_VALUE) for _ in x_mean]
        }
        with open(output_dirpath + 'standardization.json', 'w') as f:
            json.dump(standardized_to_json, f)


        normed_sig_hlf = normed_sig_train_frame.values
        normed_sig_test_hlf = normed_sig_test_frame.values

        column_list = [col_name for col_name in normed_sig_test_frame.columns]
        hlf_vars_columns = {col_name: i for i, col_name in enumerate(column_list)}

        normed_bkg_hlf = normed_bkg_train_frame.values
        normed_bkg_test_hlf = normed_bkg_test_frame.values

        sig_label = np.ones(len(normed_sig_hlf))
        bkg_label = np.zeros(len(normed_bkg_hlf))
        sig_test_label = np.ones(len(normed_sig_test_hlf))
        bkg_test_label = np.zeros(len(normed_bkg_test_hlf))

        # Build train data arrays
        data_hlf = np.concatenate((normed_sig_hlf, normed_bkg_hlf))
        label = np.concatenate((sig_label, bkg_label))
        # Shuffle train arrays
        p = rng.permutation(len(data_hlf))
        data_hlf, label = data_hlf[p], label[p]
        # Build and shuffle train DFs
        data_df = pd.concat([sig_train_frame, bkg_train_frame], ignore_index=True)
        data_df = (data_df.reindex(p)).reset_index(drop=True)
        data_aux = pd.concat([sig_aux_train_frame, bkg_aux_train_frame], ignore_index=True)
        data_aux = (data_aux.reindex(p)).reset_index(drop=True)
        print("Data HLF: {}".format(data_hlf.shape))
        print(f"n signal = {len(label[label == 1])}, n bkg = {len(label[label == 0])}")

        # Build test data arrays
        data_hlf_test = np.concatenate((normed_sig_test_hlf, normed_bkg_test_hlf))
        label_test = np.concatenate((sig_test_label, bkg_test_label))
        # Shuffle test arrays
        p_test = rng.permutation(len(data_hlf_test))
        data_hlf_test, label_test = data_hlf_test[p_test], label_test[p_test]
        # Build and shuffle test DFs
        data_test_df = pd.concat([sig_test_frame, bkg_test_frame], ignore_index=True)
        data_test_df = (data_test_df.reindex(p_test)).reset_index(drop=True)
        data_test_aux = pd.concat([sig_aux_test_frame, bkg_aux_test_frame], ignore_index=True)
        data_test_aux = (data_test_aux.reindex(p_test)).reset_index(drop=True)
        print("Data HLF test: {}".format(data_hlf_test.shape))
        print(f"n signal = {len(label_test[label_test == 1])}, n bkg = {len(label_test[label_test == 0])}")

        if not k_fold_test:
            return (
                data_df, data_test_df, 
                data_hlf, label, 
                data_hlf_test, label_test, 
                high_level_fields, high_level_fields, hlf_vars_columns,
                data_aux, data_test_aux
            )
        elif k_fold_test and fold == 0:
            (
                full_data_df, full_data_test_df, 
                full_data_hlf, full_label, 
                full_data_hlf_test, full_label_test, 
                full_high_level_fields, full_input_hlf_vars, full_hlf_vars_columns,
                full_data_aux, full_data_test_aux
            ) = (
                {f'fold_{0}': copy.deepcopy(data_df)}, {f'fold_{0}': copy.deepcopy(data_test_df)}, 
                {f'fold_{0}': copy.deepcopy(data_hlf)}, {f'fold_{0}': copy.deepcopy(label)}, 
                {f'fold_{0}': copy.deepcopy(data_hlf_test)}, {f'fold_{0}': copy.deepcopy(label_test)}, 
                {f'fold_{0}': copy.deepcopy(high_level_fields)}, {f'fold_{0}': copy.deepcopy(high_level_fields)}, {f'fold_{0}': copy.deepcopy(hlf_vars_columns)},
                {f'fold_{0}': copy.deepcopy(data_aux)}, {f'fold_{0}': copy.deepcopy(data_test_aux)}
            )
        else:
            (
                full_data_df[f'fold_{fold}'], full_data_test_df[f'fold_{fold}'], 
                full_data_hlf[f'fold_{fold}'], full_label[f'fold_{fold}'], 
                full_data_hlf_test[f'fold_{fold}'], full_label_test[f'fold_{fold}'], 
                full_high_level_fields[f'fold_{fold}'], full_input_hlf_vars[f'fold_{fold}'], full_hlf_vars_columns[f'fold_{fold}'],
                full_data_aux[f'fold_{fold}'], full_data_test_aux[f'fold_{fold}']
            ) = (
                copy.deepcopy(data_df), copy.deepcopy(data_test_df), 
                copy.deepcopy(data_hlf), copy.deepcopy(label), 
                copy.deepcopy(data_hlf_test), copy.deepcopy(label_test), 
                copy.deepcopy(high_level_fields), copy.deepcopy(high_level_fields), copy.deepcopy(hlf_vars_columns),
                copy.deepcopy(data_aux), copy.deepcopy(data_test_aux)
            )

            if k_fold_test and fold == (mod_vals[0] - 1):
                return (
                    full_data_df, full_data_test_df, 
                    full_data_hlf, full_label, 
                    full_data_hlf_test, full_label_test, 
                    full_high_level_fields, full_input_hlf_vars, full_hlf_vars_columns,
                    full_data_aux, full_data_test_aux
                )

            


