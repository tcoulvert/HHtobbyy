# Stdlib packages
import copy
import glob
import re

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak

# import matplotlib.pyplot as plt

def data_list_index_map(variable_name):
    # Order of these ifs is important b/c 'lepton' contains 'pt', so if you don't check 'pt' last there will be a bug.
    if re.search('phi', variable_name) is not None:
        index3 = 2
    elif re.search('eta', variable_name) is not None:
        index3 = 1
    else:
        index3 = 0

    # Order of these ifs is important b/c diphoton is only called 'pt' or 'eta', so it has to be checked last.
    if re.search('lepton', variable_name) is not None:
        if re.search('1', variable_name) is not None:
            index2 = 0
        else:
            index2 = 1
    elif re.search('MET', variable_name) is not None:
        index2 = 3
    else:
        index2 = 2
    
    return index2, index3

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
    if re.search('base_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            # 'abs_CosThetaStar_CS', 'abs_CosThetaStar_jj', # angular variables
            'CosThetaStar_CS','CosThetaStar_jj',
        }
    elif re.search('extra_vars', output_dirpath) is not None:
        high_level_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', 'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'DeltaR_jg_min', 'n_jets', 'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            # 'abs_CosThetaStar_CS', 'abs_CosThetaStar_jj', # angular variables
            'CosThetaStar_CS','CosThetaStar_jj',
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
            'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
            'subleadBjet_leadLepton', 'subleadBjet_subleadLepton',
        }
    else:
        raise Exception("Currently must use either base_vars of extra_vars.")
    
    improper_fill_values = {
        'leadBjet_leadLepton', 'leadBjet_subleadLepton', # deltaR btwn bjets and leptons (b/c b often decays to muons)
        'subleadBjet_leadLepton', 'subleadBjet_subleadLepton'
    }

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
        for field in improper_fill_values:
            pandas_samples[sample_name][field] = np.where(pandas_samples[sample_name][field] < 10, pandas_samples[sample_name][field], -999)

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
    def apply_log(df):
        log_fields = {
            'puppiMET_sumEt', 'puppiMET_pt', # MET variables
            'chi_t0', 'chi_t1', # jet variables
            'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
            'dijet_mass', # mass of b-dijet (resonance for H->bb)
        }
        for field in log_fields:
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

    # Standardize background
    normed_bkg_train_frame = apply_log(copy.deepcopy(bkg_train_frame))
    normed_bkg_train = (np.ma.array(normed_bkg_train_frame, mask=(normed_bkg_train_frame == FILL_VALUE)) - x_mean)/x_std
    normed_bkg_test_frame = apply_log(copy.deepcopy(bkg_test_frame))
    normed_bkg_test = (np.ma.array(normed_bkg_test_frame, mask=(normed_bkg_test_frame == FILL_VALUE)) - x_mean)/x_std
    normed_bkg_train_frame = pd.DataFrame(normed_bkg_train.filled(0), columns=list(bkg_train_frame))
    normed_bkg_test_frame = pd.DataFrame(normed_bkg_test.filled(0), columns=list(bkg_test_frame))

    # Standardize signal
    normed_sig_train_frame = apply_log(copy.deepcopy(sig_train_frame))
    normed_sig_train = (np.ma.array(normed_sig_train_frame, mask=(normed_sig_train_frame == 0)) - x_mean)/x_std
    normed_sig_test_frame = apply_log(copy.deepcopy(sig_test_frame))
    normed_sig_test = (np.ma.array(normed_sig_test_frame, mask=(normed_sig_test_frame == 0)) - x_mean)/x_std
    normed_sig_train_frame = pd.DataFrame(normed_sig_train.filled(0), columns=list(sig_train_frame))
    normed_sig_test_frame = pd.DataFrame(normed_sig_test.filled(0), columns=list(sig_test_frame))

    def to_p_list(data_frame):
        # Inputs: Pandas data frame
        # Outputs: Numpy array of dimension (Event, Particle, Attributes)
        
        particle_list_sig = np.zeros(shape=(len(data_frame), n_particles, n_particle_fields))
        # 4: max particles: l1, l2, dipho, MET
        # 6: pt, eta, phi, isLep, isDipho, isMET

        for var_idx, var_name in enumerate(['lepton1', 'lepton2', '', 'puppiMET']):
            if var_name != '':
                var_name = var_name + '_'
            particle_list_sig[:, var_idx, 0] = data_frame[var_name+'pt']
            particle_list_sig[:, var_idx, 1] = np.where(data_frame[var_name+'pt'].to_numpy() != 0, data_frame[var_name+'eta'], 0)
            particle_list_sig[:, var_idx, 2] = np.where(data_frame[var_name+'pt'].to_numpy() != 0, data_frame[var_name+'phi'], 0)
            particle_list_sig[:, var_idx, 3] = np.where(data_frame[var_name+'pt'].to_numpy() != 0, 1, 0) if re.search('lepton', var_name) is not None else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            particle_list_sig[:, var_idx, 4] = np.where(data_frame[var_name+'pt'].to_numpy() != 0, 1, 0) if re.search('lepton', var_name) is None and re.search('puppiMET', var_name) is None else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            particle_list_sig[:, var_idx, 5] = np.where(data_frame[var_name+'pt'].to_numpy() != 0, 1, 0) if re.search('puppiMET', var_name) is not None else np.zeros_like(data_frame[var_name+'pt'].to_numpy())
        # figure out how to do this without loop # -> should we even do this??
        # sorted_particle_list = np.zeros(shape=(len(data_frame), n_particles, n_particle_fields))
        # sorted_indices = np.fliplr(np.argsort(particle_list_sig[:,:,0], axis=1))
        # for i in range(len(data_frame)):
        #     sorted_particle_list[i,:,:] = particle_list_sig[i, sorted_indices[i], :]
        # nonzero_indices = np.array(np.where(sorted_particle_list[:, :, 0] != 0, True, False))
        # for i in range(len(data_frame)):
        #     sorted_particle_list[i, :np.sum(nonzero_indices[i]), :] = sorted_particle_list[i, nonzero_indices[i], :]
        #     sorted_particle_list[i, np.sum(nonzero_indices[i]):, :] = np.zeros((n_particles-np.sum(nonzero_indices[i]), n_particle_fields))
            
        # return sorted_particle_list

        return particle_list_sig
    
    normed_sig_list = to_p_list(normed_sig_train_frame)
    normed_sig_test_list = to_p_list(normed_sig_test_frame)
    normed_bkg_list = to_p_list(normed_bkg_train_frame)
    normed_bkg_test_list = to_p_list(normed_bkg_test_frame)

    if re.search('base_vars', output_dirpath) is not None:
        input_hlf_vars = [
            'puppiMET_sumEt','DeltaPhi_j1MET','DeltaPhi_j2MET','DeltaR_jg_min','n_jets','chi_t0', 'chi_t1',
            # 'abs_CosThetaStar_CS','abs_CosThetaStar_jj'
            'CosThetaStar_CS','CosThetaStar_jj',
        ]
    elif re.search('extra_vars', output_dirpath) is not None:
        input_hlf_vars = [
            'puppiMET_sumEt','DeltaPhi_j1MET','DeltaPhi_j2MET','DeltaR_jg_min','n_jets','chi_t0', 'chi_t1',
            # 'abs_CosThetaStar_CS','abs_CosThetaStar_jj',
            'CosThetaStar_CS','CosThetaStar_jj',
            'dijet_mass', 'leadBjet_leadLepton', 'leadBjet_subleadLepton', 'subleadBjet_leadLepton', 'subleadBjet_subleadLepton'
        ]
    else:
        raise Exception("Currently must use either base_vars of extra_vars.")

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
    data_df = data_df.reindex(p)
    data_aux = pd.concat([sig_aux_train_frame, bkg_aux_train_frame], ignore_index=True)
    data_aux = data_aux.reindex(p)
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
    data_test_df = data_test_df.reindex(p_test)
    data_test_aux = pd.concat([sig_aux_test_frame, bkg_aux_test_frame], ignore_index=True)
    data_test_aux = data_test_aux.reindex(p_test)
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
