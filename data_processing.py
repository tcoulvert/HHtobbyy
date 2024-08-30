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

def data_list_index_map(variable_name, data_list, event_mask):
    # Order of these ifs is important b/c 'lepton' contains 'pt', so if you don't check 'pt' last there will be a bug.
    if re.search('phi', variable_name) is not None:
        index3 = 2
    elif re.search('eta', variable_name) is not None:
        index3 = 1
    else:
        index3 = 0

    mask_arr = np.zeros_like(data_list, dtype=bool)
    if re.search('lepton', variable_name) is not None:
        if re.search('1', variable_name) is not None:
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                lepton1_idx = np.nonzero(np.where(data_list[i, :, 3] == 1, True, False))[0][0]
                mask_arr[i, lepton1_idx, index3] = True
        else:
            for i in range(len(data_list)):
                if not event_mask[i]:
                    continue
                lepton2_idx = np.nonzero(np.where(data_list[i, :, 3] == 1, True, False))[0][1]
                mask_arr[i, lepton2_idx, index3] = True
    elif re.search('MET', variable_name) is not None:
        for i in range(len(data_list)):
            if not event_mask[i]:
                continue
            MET_idx = np.nonzero(np.where(data_list[i, :, 5] == 1, True, False))[0][0]
            mask_arr[i, MET_idx, index3] = True
    else:
        for i in range(len(data_list)):
            if not event_mask[i]:
                continue
            diphoton_idx = np.nonzero(np.where(data_list[i, :, 4] == 1, True, False))[0][0]
            mask_arr[i, diphoton_idx, index3] = True
    
    return mask_arr

def process_data(n_particles, n_particle_fields, signal_filepaths, bkg_filepaths, output_dirpath, seed=None, return_pre_std=False):
    # Load parquet files #
    
    sig_samples_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in signal_filepaths]
    sig_samples_pq = ak.concatenate(sig_samples_list)
    bkg_samples_list = [ak.from_parquet(glob.glob(dir_path)) for dir_path in bkg_filepaths]
    bkg_samples_pq = ak.concatenate(bkg_samples_list)
    samples = {
        'sig': sig_samples_pq,
        'bkg': bkg_samples_pq,
    }
    
    # Convert parquet files to pandas DFs #
    pandas_samples = {}
    extra_RNN_vars = []
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
        if re.search('and_bools', output_dirpath) is not None:
            high_level_fields = high_level_fields | {
                'chi_t0_bool', 'chi_t1_bool',
                'leadBjet_leadLepton_bool', 'leadBjet_subleadLepton_bool',
                'subleadBjet_leadLepton_bool', 'subleadBjet_subleadLepton_bool'
            }
        elif re.search('in_RNN', output_dirpath) is not None:
            print('gotten into RNN if statement')
            extra_RNN_vars = [
                'chi_t0', 'chi_t1', 'leadBjet_leadLepton', 'leadBjet_subleadLepton',
                'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
            ]
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

    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'eventWeight',  # computed eventWeight using (genWeight * lumi * xs / sum_of_genWeights)
        'mass', 'dijet_mass' # diphoton and bb-dijet mass
    } # https://stackoverflow.com/questions/67003141/how-to-remove-a-field-from-a-collection-of-records-created-by-awkward-zip

    for sample_name, sample in samples.items():
        pandas_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in high_level_fields
        }
        pandas_aux_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in high_level_aux_fields
        }
        # for field in improper_fill_values:
        #     pandas_samples[sample_name][field] = np.where(pandas_samples[sample_name][field] < 10, pandas_samples[sample_name][field], -999)
        # pandas_samples[sample_name]['puppiMET_eta'] = np.zeros_like(pandas_samples[sample_name]['puppiMET_pt'])

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

    def train_test_split_df(sig_df, sig_aux_df, bkg_df, bkg_aux_df, method='modulus'):
        if method == 'modulus':
            # Train/Val events are those with odd event #s, test events have even event #s
            sig_train_frame = sig_df.loc[(sig_aux_df['event'] % 2).ne(0)].reset_index(drop=True)
            sig_test_frame = sig_df.loc[(sig_aux_df['event'] % 2).ne(1)].reset_index(drop=True)

            sig_aux_train_frame = sig_aux_df.loc[(sig_aux_df['event'] % 2).ne(0)].reset_index(drop=True)
            sig_aux_test_frame = sig_aux_df.loc[(sig_aux_df['event'] % 2).ne(1)].reset_index(drop=True)

            bkg_train_frame = bkg_df.loc[(bkg_aux_df['event'] % 2).ne(0)].reset_index(drop=True)
            bkg_test_frame = bkg_df.loc[(bkg_aux_df['event'] % 2).ne(1)].reset_index(drop=True)

            bkg_aux_train_frame = bkg_aux_df.loc[(bkg_aux_df['event'] % 2).ne(0)].reset_index(drop=True)
            bkg_aux_test_frame = bkg_aux_df.loc[(bkg_aux_df['event'] % 2).ne(1)].reset_index(drop=True)
        else:
            raise Exception(f"Only 2 accepted methods: 'sample' and 'modulus'. You input {method}")
        return sig_train_frame, sig_test_frame, sig_aux_train_frame, sig_aux_test_frame, bkg_train_frame, bkg_test_frame, bkg_aux_train_frame, bkg_aux_test_frame

    (
        sig_train_frame, sig_test_frame, 
        sig_aux_train_frame, sig_aux_test_frame, 
        bkg_train_frame, bkg_test_frame,
        bkg_aux_train_frame, bkg_aux_test_frame
    ) = train_test_split_df(sig_frame, sig_aux_frame, bkg_frame, bkg_aux_frame)


    # Perform the standardization #
    no_standardize = {
        'puppiMET_eta', 'puppiMET_phi', # MET variables
        'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
        'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
        'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
        'CosThetaStar_CS','CosThetaStar_jj',
        'chi_t0_bool', 'chi_t1_bool',
        'leadBjet_leadLepton_bool', 'leadBjet_subleadLepton_bool',
        'subleadBjet_leadLepton_bool', 'subleadBjet_subleadLepton_bool'
    }
    def apply_log(df):
        log_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', # MET variables
            'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
        }
        for field in log_fields & high_level_fields:
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
    bkg_train_min = np.min(normed_bkg_train, axis=0)
    normed_bkg_train_frame = pd.DataFrame(normed_bkg_train.filled(FILL_VALUE), columns=list(bkg_train_frame))
    normed_bkg_test_frame = pd.DataFrame(normed_bkg_test.filled(FILL_VALUE), columns=list(bkg_test_frame))

    # Standardize signal
    normed_sig_train_frame = apply_log(copy.deepcopy(sig_train_frame))
    normed_sig_train = (np.ma.array(normed_sig_train_frame, mask=(normed_sig_train_frame == FILL_VALUE)) - x_mean)/x_std
    normed_sig_test_frame = apply_log(copy.deepcopy(sig_test_frame))
    normed_sig_test = (np.ma.array(normed_sig_test_frame, mask=(normed_sig_test_frame == FILL_VALUE)) - x_mean)/x_std
    sig_train_min = np.min(normed_sig_train, axis=0)
    normed_sig_train_frame = pd.DataFrame(normed_sig_train.filled(FILL_VALUE), columns=list(sig_train_frame))
    normed_sig_test_frame = pd.DataFrame(normed_sig_test.filled(FILL_VALUE), columns=list(sig_test_frame))

    train_min = np.min(np.vstack((sig_train_min, bkg_train_min)), axis=0)
    train_min_mean = np.mean(np.vstack((train_min, -10*np.ones_like(train_min))), axis=0)
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
                train_min_mean[i]
            )
    standardized_to_json = {
        'standardized_variables': [col for col in df_train.columns],
        'standardized_mean': [mean for mean in x_mean],
        'standardized_stddev': [std for std in x_std],
        'standardized_unphysical_values': [min_mean for min_mean in train_min_mean]
    }
    with open(output_dirpath + 'standardization.json', 'w') as f:
        json.dump(standardized_to_json, f)

    def to_p_list(data_frame):
        # Inputs: Pandas data frame
        # Outputs: Numpy array of dimension (Event, Particle, Attributes)
        
        particle_list_sig = np.zeros(shape=(len(data_frame), n_particles+len(extra_RNN_vars), n_particle_fields+len(extra_RNN_vars)))  # +(1 if len(extra_RNN_vars) > 0 else 0)
        # 4: max particles: l1, l2, dipho, MET
        # 6: pt, eta, phi, isLep, isDipho, isMET

        for var_idx, var_name in enumerate(['lepton1', 'lepton2', '', 'puppiMET']):
            if var_name != '':
                var_name = var_name + '_'
            particle_list_sig[:, var_idx, 0] = np.where(data_frame[var_name+'pt'].to_numpy() != train_min_mean[col_idx_dict[var_name+'pt']], data_frame[var_name+'pt'].to_numpy(), 0)
            particle_list_sig[:, var_idx, 1] = np.where(data_frame[var_name+'eta'].to_numpy() != train_min_mean[col_idx_dict[var_name+'eta']], data_frame[var_name+'eta'].to_numpy(), 0)
            particle_list_sig[:, var_idx, 2] = np.where(data_frame[var_name+'phi'].to_numpy() != train_min_mean[col_idx_dict[var_name+'phi']], data_frame[var_name+'phi'].to_numpy(), 0)
            particle_list_sig[:, var_idx, 3] = np.ones_like(data_frame[var_name+'pt'].to_numpy()) if re.search('lepton', var_name) is not None else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            particle_list_sig[:, var_idx, 4] = np.ones_like(data_frame[var_name+'pt'].to_numpy()) if len(var_name) == 0 else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            particle_list_sig[:, var_idx, 5] = np.ones_like(data_frame[var_name+'pt'].to_numpy()) if re.search('puppiMET', var_name) is not None else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            for i in range(len(extra_RNN_vars)):
                particle_list_sig[:, var_idx, 6+i] = np.zeros_like(data_frame[var_name+'pt'].to_numpy())
        for var_idx, var_name in enumerate(extra_RNN_vars, start=4):
            particle_list_sig[:, var_idx, 0] = np.where(data_frame[var_name].to_numpy() != train_min_mean[col_idx_dict[var_name]], data_frame[var_name].to_numpy(), 0)
            particle_list_sig[:, var_idx, 1:] = np.zeros_like(particle_list_sig[:, 0, 1:])
            particle_list_sig[:, var_idx, 2+var_idx] = np.ones_like(particle_list_sig[:, var_idx, 0])

        # Sort the particles in each event in the particle_list
        #   -> this sorting is used later on to tell the RNN which particles to drop in each event
        sorted_particle_list = np.zeros(shape=(len(data_frame), n_particles, n_particle_fields))
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
        'puppiMET_sumEt','DeltaR_jg_min','n_jets','chi_t0', 'chi_t1',
        'CosThetaStar_CS','CosThetaStar_jj', 'DeltaPhi_j1MET','DeltaPhi_j2MET',
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

    if return_pre_std:
        return (
            data_df, data_test_df, 
            data_list, data_hlf, label, 
            data_list_test, data_hlf_test, label_test, 
            high_level_fields, input_hlf_vars, hlf_vars_columns,
            data_aux, data_test_aux
        )
    else:
        return (
            data_list, data_hlf, label, 
            data_list_test, data_hlf_test, label_test, 
            high_level_fields, input_hlf_vars, hlf_vars_columns,
            data_aux, data_test_aux
        )
