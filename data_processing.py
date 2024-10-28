# Stdlib packages
import copy
import glob
import json
import os
import re

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

import matplotlib.pyplot as plt

def data_list_index_map(variable_name, data_list, event_mask, n_pFields=7):
    # Order of these ifs is important b/c 'lepton' contains 'pt', so if you don't check 'pt' last there will be a bug.
    if re.search('phi', variable_name) is not None:
        index3 = 2
    elif re.search('eta', variable_name) is not None:
        index3 = 1
    elif (
        (re.search('subleadBjet', variable_name) is not None) 
        or (re.search('j2MET', variable_name) is not None)
    ) and n_pFields == 9:
        index3 = 4
    elif (
        (re.search('leadBjet', variable_name) is not None)
        or (re.search('j1MET', variable_name) is not None)
    ) and n_pFields == 9:
        index3 = 3
    elif (re.search('DeltaR_jg_min', variable_name) is not None) and n_pFields == 9:
        index3 = [4, 5]
    else:
        index3 = 0

    mask_arr = np.zeros_like(data_list, dtype=bool)
    if re.search('epton', variable_name) is not None:
        if (re.search('2', variable_name) is not None) or (re.search('subleadLepton', variable_name) is not None):
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                lepton2_idx = np.nonzero(np.where(data_list[i, :, n_pFields-4] == 1, True, False))[0][1]
                mask_arr[i, lepton2_idx, index3] = True
        else:
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                lepton1_idx = np.nonzero(np.where(data_list[i, :, n_pFields-4] == 1, True, False))[0][0]
                mask_arr[i, lepton1_idx, index3] = True
    elif re.search('MET', variable_name) is not None:
        for i in range(len(data_list)):
            if not event_mask[i]:
                continue
            MET_idx = np.nonzero(np.where(data_list[i, :, n_pFields-2] == 1, True, False))[0][0]
            mask_arr[i, MET_idx, index3] = True
    elif re.search('bjet', variable_name) is not None:
        if re.search('sublead', variable_name) is not None:
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                bjet2_idx = np.nonzero(np.where(data_list[i, :, n_pFields-1] == 1, True, False))[0][1]
                mask_arr[i, bjet2_idx, index3] = True
        else:
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                bjet1_idx = np.nonzero(np.where(data_list[i, :, n_pFields-1] == 1, True, False))[0][0]
                mask_arr[i, bjet1_idx, index3] = True
    else:
        for i in range(len(data_list)):
            for index3_ in (index3 if variable_name == 'DeltaR_jg_min' and n_pFields == 9 else [index3]):
                if not event_mask[i]:
                    continue
                diphoton_idx = np.nonzero(np.where(data_list[i, :, n_pFields-3] == 1, True, False))[0][0]
                mask_arr[i, diphoton_idx, index3_] = True
    
    return mask_arr

def process_data(
    n_particles, n_particle_fields, 
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
    for sample in samples.values():
        print(sample['n_leptons'])
        sample['n_leptons'] = ak.where(sample['n_leptons'] == -999, ak.zeros_like(sample['n_leptons']), sample['n_leptons'])
        print(sample['n_leptons'])
    
    # Convert parquet files to pandas DFs #
    pandas_samples = {}
    extra_RNN_vars = []
    dont_include_vars = []
    if re.search('base_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
        }
    elif re.search('extra_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
            'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
            'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
        }
        if re.search('no_dijet_mass', output_dirpath) is not None:
            high_level_fields.remove('dijet_mass')
        if re.search('lead_lep_only', output_dirpath) is not None:
            dont_include_vars = [
                'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
                'leadBjet_subleadLepton', 'subleadBjet_subleadLepton'
            ]
        elif re.search('no_lep', output_dirpath) is not None:
            dont_include_vars = [
                'lepton1_pt', 'lepton1_eta', 'lepton1_phi',
                'leadBjet_leadLepton', 'subleadBjet_leadLepton',
                'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 
                'leadBjet_subleadLepton', 'subleadBjet_subleadLepton'
            ]
        if re.search('and_bools', output_dirpath) is not None:
            high_level_fields = high_level_fields | {
                'chi_t0_bool', 'chi_t1_bool',
                'leadBjet_leadLepton_bool', 'leadBjet_subleadLepton_bool',
                'subleadBjet_leadLepton_bool', 'subleadBjet_subleadLepton_bool'
            }
        elif re.search('in_RNN', output_dirpath) is not None:
            extra_RNN_vars = [
                'chi_t0', 'chi_t1', 'leadBjet_leadLepton', 'leadBjet_subleadLepton',
                'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
            ]
        if re.search('\+', output_dirpath) is not None:
            high_level_fields = high_level_fields | {
                'n_leptons', 
                'lead_bjet_pt', 'lead_bjet_eta', 'lead_bjet_phi',
                'sublead_bjet_pt', 'sublead_bjet_eta', 'sublead_bjet_phi',
            }
    elif re.search('no_bad_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
        }
        if re.search('no_dijet_mass', output_dirpath) is not None:
            high_level_fields.remove('dijet_mass')
    elif re.search('simplified_bad_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',  # angular variables
            'dijet_mass', # mass of b-dijet (resonance for H->bb),
            'n_leptons'
        }
        if re.search('no_dijet_mass', output_dirpath) is not None:
            high_level_fields.remove('dijet_mass')
    else:
        raise Exception("Currently must use either base_vars of extra_vars.")
    extra_RNN_vars.sort()
    dont_include_vars.sort()


    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'eventWeight',  # computed eventWeight using (genWeight * lumi * xs / sum_of_genWeights)
        'mass', 'dijet_mass', # diphoton and bb-dijet mass
        'lepton1_pt', 'lepton2_pt',
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

            sig_train_slice = (sig_train_frame['lepton1_pt'] == -999)
            bkg_train_slice = (bkg_train_frame['lepton1_pt'] == -999)
            sig_train_frame = sig_train_frame.loc[sig_train_slice, keep_cols].reset_index(drop=True)
            bkg_train_frame = bkg_train_frame.loc[bkg_train_slice, keep_cols].reset_index(drop=True)
            sig_aux_train_frame = sig_aux_train_frame.loc[sig_train_slice].reset_index(drop=True)
            bkg_aux_train_frame = bkg_aux_train_frame.loc[bkg_train_slice].reset_index(drop=True)

            # sig_test_slice = (sig_test_frame['lepton1_pt'] > -1000)
            # bkg_test_slice = (bkg_test_frame['lepton1_pt'] > -1000)
            sig_test_frame = sig_test_frame[keep_cols].reset_index(drop=True)
            bkg_test_frame = bkg_test_frame[keep_cols].reset_index(drop=True)
            # sig_aux_test_frame = sig_aux_test_frame.loc[sig_test_slice].reset_index(drop=True)
            # bkg_aux_test_frame = bkg_aux_test_frame.loc[bkg_test_slice].reset_index(drop=True)

            for lepton2_var in dont_include_vars:
                high_level_fields.remove(lepton2_var)
        ## End further selection for lepton-veto check ##


        # Perform the standardization #
        no_standardize = {
            'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',
            'chi_t0_bool', 'chi_t1_bool',
            'leadBjet_leadLepton_bool', 'leadBjet_subleadLepton_bool',
            'subleadBjet_leadLepton_bool', 'subleadBjet_subleadLepton_bool',
            'lead_bjet_eta', 'lead_bjet_phi',
            'sublead_bjet_eta', 'sublead_bjet_phi',
        }
        log_fields = {
            'puppiMET_sumEt', 
            'puppiMET_pt', # MET variables
            'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lead_bjet_pt', 'sublead_bjet_pt', # bjet pts
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
        }
        def apply_log(df):
            for field in (log_fields & high_level_fields) - no_standardize:
                df[field] = np.where(df[field] > 0, np.log(df[field]), df[field])
            return df
        FILL_VALUE = -999
        # Because of zero-padding, standardization needs special treatment
        df_train = pd.concat([sig_train_frame, bkg_train_frame], ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_train = apply_log(copy.deepcopy(df_train))
        masked_x_sample = np.ma.array(df_train, mask=(df_train == FILL_VALUE))
        x_mean = masked_x_sample.mean(axis=0)
        x_std = masked_x_sample.std(axis=0)
        for i, col in enumerate(df_train.columns):
            if col in no_standardize:
                x_mean[i] = 0
                x_std[i] = 1

        # Standardize background
        normed_bkg_train_frame = apply_log(copy.deepcopy(bkg_train_frame))
        normed_bkg_train = (np.ma.array(normed_bkg_train_frame, mask=(normed_bkg_train_frame == FILL_VALUE)) - x_mean)/x_std
        normed_bkg_test_frame = apply_log(copy.deepcopy(bkg_test_frame))
        normed_bkg_test = (np.ma.array(normed_bkg_test_frame, mask=(normed_bkg_test_frame == FILL_VALUE)) - x_mean)/x_std
        # bkg_train_min = np.min(normed_bkg_train, axis=0)
        bkg_train_max = np.max(normed_bkg_train, axis=0)
        normed_bkg_train_frame = pd.DataFrame(normed_bkg_train.filled(FILL_VALUE), columns=list(bkg_train_frame))
        normed_bkg_test_frame = pd.DataFrame(normed_bkg_test.filled(FILL_VALUE), columns=list(bkg_test_frame))

        # Standardize signal
        normed_sig_train_frame = apply_log(copy.deepcopy(sig_train_frame))
        normed_sig_train = (np.ma.array(normed_sig_train_frame, mask=(normed_sig_train_frame == FILL_VALUE)) - x_mean)/x_std
        normed_sig_test_frame = apply_log(copy.deepcopy(sig_test_frame))
        normed_sig_test = (np.ma.array(normed_sig_test_frame, mask=(normed_sig_test_frame == FILL_VALUE)) - x_mean)/x_std
        # sig_train_min = np.min(normed_sig_train, axis=0)
        sig_train_max = np.max(normed_sig_train, axis=0)
        normed_sig_train_frame = pd.DataFrame(normed_sig_train.filled(FILL_VALUE), columns=list(sig_train_frame))
        normed_sig_test_frame = pd.DataFrame(normed_sig_test.filled(FILL_VALUE), columns=list(sig_test_frame))

        # train_min = np.min(np.vstack((sig_train_min, bkg_train_min)), axis=0)
        # train_pad = np.mean(np.vstack((train_min, -10*np.ones_like(train_min))), axis=0)
        train_max = np.min(np.vstack((sig_train_max, bkg_train_max)), axis=0)
        train_pad = np.mean(np.vstack((train_max, 10*np.ones_like(train_max))), axis=0)
        col_idx_dict = {col: i for i, col in enumerate(df_train.columns)}
        for df in [
            normed_sig_train_frame, normed_sig_test_frame, normed_bkg_train_frame, normed_bkg_test_frame
        ]:
            for col, i in col_idx_dict.items():
                if np.all(df[col] != FILL_VALUE):
                    continue
                df[col] = np.where(
                    df[col] != FILL_VALUE, 
                    df[col], 
                    train_pad[i]
                )
        standardized_to_json = {
            'standardized_logs': [True if col in log_fields else False for col in df_train.columns],
            'standardized_variables': [col for col in df_train.columns],
            'standardized_mean': [float(mean) for mean in x_mean],
            'standardized_stddev': [float(std) for std in x_std],
            'standardized_unphysical_values': [float(min_mean) for min_mean in train_pad]
        }
        with open(os.path.join(output_dirpath, f'ttH_Killer_IN_{fold}_standardization.json'), 'w') as f:
            json.dump(standardized_to_json, f)

        def to_p_list(data_frame):
            # Inputs: Pandas data frame
            # Outputs: Numpy array of dimension (Event, Particle, Attributes)
            
            particle_list_sig = np.zeros(shape=(len(data_frame), n_particles+len(extra_RNN_vars), n_particle_fields+len(extra_RNN_vars)))  # +(1 if len(extra_RNN_vars) > 0 else 0)
            # 4: max particles: l1, l2, dipho, MET
            # 6: pt, eta, phi, isLep, isDipho, isMET
            if n_particles == 4:
                var_names = ['lepton1', 'lepton2', '', 'puppiMET']
                var_one_hots = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            elif n_particles == 3:
                var_names = ['lepton1', '', 'puppiMET']
                var_one_hots = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            elif n_particles == 2:
                var_names = ['', 'puppiMET']
                var_one_hots = [[1, 0], [0, 1]]
            elif n_particles == 6:
                var_names = ['lepton1', 'lepton2', '', 'puppiMET', 'lead_bjet', 'sublead_bjet']
                var_one_hots = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
            else:
                raise Exception(f"Currently the only supported n_particles are 3 and 4. You passed in {n_particles}.")
            
            data_types = {0: 'pt', 1: 'eta', 2: 'phi'}
            # data_types = {0: 'pt', 1: 'eta', 2: 'phi', 3: 'j1', 4: 'j2'}

            for var_idx, var_name in enumerate(var_names):
                if var_name != '':
                    var_name = var_name + '_'

                for local_idx, data_type in data_types.items():
                    if data_type in {'pt', 'eta', 'phi'}:
                        particle_list_sig[:, var_idx, local_idx] = np.where(data_frame[var_name+data_type].to_numpy() != train_pad[col_idx_dict[var_name+data_type]], data_frame[var_name+data_type].to_numpy(), 0)
                    elif re.search('lepton', var_name) is not None:  # (sub)leadBjet_(sub)leadLepton
                        data_type_ = ('' if data_type == 'j1' else 'sub') + 'leadBjet_' + ('' if re.search('1', var_name) is not None else 'sub') + 'leadLepton'
                        particle_list_sig[:, var_idx, local_idx] = np.where(data_frame[data_type_].to_numpy() != train_pad[col_idx_dict[data_type_]], data_frame[data_type_].to_numpy(), 0)
                    elif re.search('MET', var_name) is not None:  # DeltaPhi_j1MET
                        data_type_ = 'DeltaPhi_' + data_type + 'MET'
                        particle_list_sig[:, var_idx, local_idx] = np.where(data_frame[data_type_].to_numpy() != train_pad[col_idx_dict[data_type_]], data_frame[data_type_].to_numpy(), 0)
                    elif re.search('bjet', var_name) is not None:  # np.zeros()
                        particle_list_sig[:, var_idx, local_idx] = np.zeros_like(data_frame[var_name+'pt'].to_numpy())
                    else:  # diphoton
                        data_type_ = 'DeltaR_jg_min'
                        particle_list_sig[:, var_idx, local_idx] = np.where(data_frame[data_type_].to_numpy() != train_pad[col_idx_dict[data_type_]], data_frame[data_type_].to_numpy(), 0)
                    
                
                particle_list_sig[:, var_idx, len(data_types):] = np.tile(var_one_hots[var_idx], (data_frame[var_name+'pt'].shape[0], 1))

            # Sort the particles in each event in the particle_list
            #   -> this sorting is used later on to tell the RNN which particles to drop in each event
            sorted_particle_list = np.zeros_like(particle_list_sig)
            sorted_indices = np.fliplr(np.argsort(particle_list_sig[:,:,0], axis=1))
            
            for i in range(len(data_frame)):
                sorted_particle_list[i,:,:] = particle_list_sig[i, sorted_indices[i], :]
            
            nonzero_indices = np.array(np.where(sorted_particle_list[:,:,0] != 0, True, False))
            zero_indices = np.logical_not(nonzero_indices)
            
            for i in range(len(data_frame)):
                copy_arr = copy.deepcopy(sorted_particle_list[i, zero_indices[i], :])
                sorted_particle_list[i, :np.sum(nonzero_indices[i]), :] = sorted_particle_list[i, nonzero_indices[i], :]
                sorted_particle_list[i, np.sum(nonzero_indices[i]):, :] = copy.deepcopy(copy_arr)
                
            return sorted_particle_list
        
        normed_sig_list = to_p_list(normed_sig_train_frame)
        normed_sig_test_list = to_p_list(normed_sig_test_frame)
        normed_bkg_list = to_p_list(normed_bkg_train_frame)
        normed_bkg_test_list = to_p_list(normed_bkg_test_frame)

        input_hlf_vars_max = [
            'puppiMET_sumEt',
            'n_jets','chi_t0', 'chi_t1',
            'CosThetaStar_CS','CosThetaStar_jj', 
            'DeltaR_jg_min',
            'DeltaPhi_j1MET','DeltaPhi_j2MET',
            'leadBjet_leadLepton', 'leadBjet_subleadLepton', 'subleadBjet_leadLepton', 'subleadBjet_subleadLepton', 
            'dijet_mass',
            'chi_t0_bool', 'chi_t1_bool',
            'leadBjet_leadLepton_bool', 'leadBjet_subleadLepton_bool', 'subleadBjet_leadLepton_bool', 'subleadBjet_subleadLepton_bool',
            'n_leptons'
        ]
        input_hlf_vars = []
        for var in input_hlf_vars_max:
            if var in high_level_fields and var not in set(extra_RNN_vars):
                input_hlf_vars.append(var)
        input_hlf_vars.sort()

        normed_sig_hlf = normed_sig_train_frame[input_hlf_vars].values
        normed_sig_test_hlf = normed_sig_test_frame[input_hlf_vars].values

        column_list = [col_name for col_name in normed_sig_test_frame[input_hlf_vars].columns]
        hlf_vars_columns = {col_name: i for i, col_name in enumerate(column_list)}

        normed_bkg_hlf = normed_bkg_train_frame[input_hlf_vars].values
        normed_bkg_test_hlf = normed_bkg_test_frame[input_hlf_vars].values

        sig_label = np.ones(len(normed_sig_hlf))
        bkg_label = np.zeros(len(normed_bkg_hlf))
        sig_test_label = np.ones(len(normed_sig_test_hlf))
        bkg_test_label = np.zeros(len(normed_bkg_test_hlf))

        # Build train data arrays
        data_list = np.concatenate((normed_sig_list, normed_bkg_list))
        data_hlf = np.concatenate((normed_sig_hlf, normed_bkg_hlf))
        label = np.concatenate((sig_label, bkg_label))
        # Shuffle train arrays
        p = rng.permutation(len(data_list))
        data_list, data_hlf, label = data_list[p], data_hlf[p], label[p]
        # Build and shuffle train DFs
        data_df = pd.concat([sig_train_frame, bkg_train_frame], ignore_index=True)
        data_df = (data_df.reindex(p)).reset_index(drop=True)
        data_aux = pd.concat([sig_aux_train_frame, bkg_aux_train_frame], ignore_index=True)
        data_aux = (data_aux.reindex(p)).reset_index(drop=True)
        print("Data list: {}".format(data_list.shape))
        print("Data HLF: {}".format(data_hlf.shape))
        print(f"n signal = {len(label[label == 1])}, n bkg = {len(label[label == 0])}")

        # Build test data arrays
        data_list_test = np.concatenate((normed_sig_test_list, normed_bkg_test_list))
        data_hlf_test = np.concatenate((normed_sig_test_hlf, normed_bkg_test_hlf))
        label_test = np.concatenate((sig_test_label, bkg_test_label))
        # Shuffle test arrays
        p_test = rng.permutation(len(data_list_test))
        data_list_test, data_hlf_test, label_test = data_list_test[p_test], data_hlf_test[p_test], label_test[p_test]
        # Build and shuffle test DFs
        data_test_df = pd.concat([sig_test_frame, bkg_test_frame], ignore_index=True)
        data_test_df = (data_test_df.reindex(p_test)).reset_index(drop=True)
        data_test_aux = pd.concat([sig_aux_test_frame, bkg_aux_test_frame], ignore_index=True)
        data_test_aux = (data_test_aux.reindex(p_test)).reset_index(drop=True)
        print("Data list test: {}".format(data_list_test.shape))
        print("Data HLF test: {}".format(data_hlf_test.shape))
        print(f"n signal = {len(label_test[label_test == 1])}, n bkg = {len(label_test[label_test == 0])}")

        if not k_fold_test:
            return (
                data_df, data_test_df, 
                data_list, data_hlf, label, 
                data_list_test, data_hlf_test, label_test, 
                high_level_fields, input_hlf_vars, hlf_vars_columns,
                data_aux, data_test_aux
            )
        elif k_fold_test and fold == 0:
            (
                full_data_df, full_data_test_df, 
                full_data_list, full_data_hlf, full_label, 
                full_data_list_test, full_data_hlf_test, full_label_test, 
                full_high_level_fields, full_input_hlf_vars, full_hlf_vars_columns,
                full_data_aux, full_data_test_aux
            ) = (
                {f'fold_{0}': copy.deepcopy(data_df)}, {f'fold_{0}': copy.deepcopy(data_test_df)}, 
                {f'fold_{0}': copy.deepcopy(data_list)}, {f'fold_{0}': copy.deepcopy(data_hlf)}, {f'fold_{0}': copy.deepcopy(label)}, 
                {f'fold_{0}': copy.deepcopy(data_list_test)}, {f'fold_{0}': copy.deepcopy(data_hlf_test)}, {f'fold_{0}': copy.deepcopy(label_test)}, 
                {f'fold_{0}': copy.deepcopy(high_level_fields)}, {f'fold_{0}': copy.deepcopy(input_hlf_vars)}, {f'fold_{0}': copy.deepcopy(hlf_vars_columns)},
                {f'fold_{0}': copy.deepcopy(data_aux)}, {f'fold_{0}': copy.deepcopy(data_test_aux)}
            )
        else:
            (
                full_data_df[f'fold_{fold}'], full_data_test_df[f'fold_{fold}'], 
                full_data_list[f'fold_{fold}'], full_data_hlf[f'fold_{fold}'], full_label[f'fold_{fold}'], 
                full_data_list_test[f'fold_{fold}'], full_data_hlf_test[f'fold_{fold}'], full_label_test[f'fold_{fold}'], 
                full_high_level_fields[f'fold_{fold}'], full_input_hlf_vars[f'fold_{fold}'], full_hlf_vars_columns[f'fold_{fold}'],
                full_data_aux[f'fold_{fold}'], full_data_test_aux[f'fold_{fold}']
            ) = (
                copy.deepcopy(data_df), copy.deepcopy(data_test_df), 
                copy.deepcopy(data_list), copy.deepcopy(data_hlf), copy.deepcopy(label), 
                copy.deepcopy(data_list_test), copy.deepcopy(data_hlf_test), copy.deepcopy(label_test), 
                copy.deepcopy(high_level_fields), copy.deepcopy(input_hlf_vars), copy.deepcopy(hlf_vars_columns),
                copy.deepcopy(data_aux), copy.deepcopy(data_test_aux)
            )

            if k_fold_test and fold == (mod_vals[0] - 1):
                return (
                    full_data_df, full_data_test_df, 
                    full_data_list, full_data_hlf, full_label, 
                    full_data_list_test, full_data_hlf_test, full_label_test, 
                    full_high_level_fields, full_input_hlf_vars, full_hlf_vars_columns,
                    full_data_aux, full_data_test_aux
                )

            


