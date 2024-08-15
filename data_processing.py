# Stdlib packages
import glob
import re

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import awkward as ak


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

def process_data(signal_filepaths, bkg_filepaths, output_dirpath, seed=None, return_pre_std=False):
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
            'subleadBjet_leadLepton', 'subleadBjet_subleadLepton'
        }
    else:
        raise Exception("Currently must use either base_vars of extra_vars.")

    pandas_aux_samples = {}
    high_level_aux_fields = {
        'event', # event number
        'mass', 'dijet_mass' # diphoton and bb-dijet mass
    } # https://stackoverflow.com/questions/67003141/how-to-remove-a-field-from-a-collection-of-records-created-by-awkward-zip

    for sample_name, sample in samples.items():
        pandas_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in high_level_fields
        }
        pandas_aux_samples[sample_name] = {
            field: ak.to_numpy(sample[field], allow_missing=False) for field in high_level_aux_fields
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
    # sig_frame = sig_frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    # bkg_frame = bkg_frame.sample(frac=1, random_state=seed).reset_index(drop=True)

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
        # elif method == 'sample':
        #     sig_train_frame = sig_df.sample(frac=0.75, random_state=seed).reset_index(drop=True)
        #     sig_test_frame = sig_df.drop(sig_train_frame.index).reset_index(drop=True)
        #     bkg_train_frame = bkg_df.sample(frac=0.75, random_state=seed).reset_index(drop=True)
        #     bkg_test_frame = bkg_df.drop(bkg_train_frame.index).reset_index(drop=True)
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
    FILL_VALUE = -999
    # Because of zero-padding, standardization needs special treatment
    masked_x_sample = np.ma.array(bkg_train_frame, mask=(bkg_train_frame == FILL_VALUE))
    x_mean = masked_x_sample.mean(axis=0)
    x_std = masked_x_sample.std(axis=0)

    # Standardize background
    normed_bkg_train = (masked_x_sample - x_mean)/x_std
    normed_bkg_test = (np.ma.array(bkg_test_frame, mask=(bkg_test_frame == FILL_VALUE)) - x_mean)/x_std

    # Standardize signal
    masked_x_sample = np.ma.array(sig_train_frame, mask=(sig_train_frame == FILL_VALUE))
    normed_sig_train = (masked_x_sample - x_mean)/x_std
    normed_sig_test = (np.ma.array(sig_test_frame, mask=(sig_test_frame == FILL_VALUE)) - x_mean)/x_std

    normed_bkg_train_frame = pd.DataFrame(normed_bkg_train.filled(0), columns=list(bkg_train_frame))
    normed_bkg_train_frame.head()
    normed_bkg_test_frame = pd.DataFrame(normed_bkg_test.filled(0), columns=list(bkg_test_frame))
    normed_bkg_test_frame.head()

    normed_sig_train_frame = pd.DataFrame(normed_sig_train.filled(0), columns=list(sig_train_frame))
    normed_sig_train_frame.head()
    normed_sig_test_frame = pd.DataFrame(normed_sig_test.filled(0), columns=list(sig_test_frame))
    normed_sig_test_frame.head()

    def to_p_list(data_frame):
        # Inputs: Pandas data frame
        # Outputs: Numpy array of dimension (Event, Particle, Attributes)
        
        particle_list_sig = np.zeros(shape=(len(data_frame),4,6))
        sorted_particle_list = np.zeros(shape=(len(data_frame),4,6))
        # 4: max particles: l1, l2, dipho, MET
        # 6: pt, eta, phi, isLep, isDipho, isMET
    
        for i in range(len(data_frame)): # loop through the list of events
            ptl1 = data_frame['lepton1_pt'][i]
            ptl2 = data_frame['lepton2_pt'][i]
            ptdipho = data_frame['pt'][i]
            ptMET = data_frame['puppiMET_pt'][i]

            etal1 = data_frame['lepton1_eta'][i]
            etal2 = data_frame['lepton2_eta'][i]
            etadipho = data_frame['eta'][i]
            etaMET = data_frame['puppiMET_eta'][i]

            phil1 = data_frame['lepton1_phi'][i]
            phil2 = data_frame['lepton2_phi'][i]
            phidipho = data_frame['phi'][i]
            phiMET = data_frame['puppiMET_phi'][i]

            # list through list of particles: l1, l2, diphoton, MET
            # 0: leading lep
            particle_list_sig[i,0, 0] = ptl1
            particle_list_sig[i,0, 1] = etal1
            particle_list_sig[i,0, 2] = phil1
            particle_list_sig[i,0, 3] = 1 if ptl1 != 0 else 0 # isLep
            particle_list_sig[i,0, 4] = 0 # isDiPho
            particle_list_sig[i,0, 5] = 0 # isMET

            # 1: subleading lep
            particle_list_sig[i,1, 0] = ptl2
            particle_list_sig[i,1, 1] = etal2
            particle_list_sig[i,1, 2] = phil2
            particle_list_sig[i,1, 3] = 1 if ptl2 != 0 else 0 # isLep
            particle_list_sig[i,1, 4] = 0 # isDiPho
            particle_list_sig[i,1, 5] = 0 # isMET

            # 2: dipho
            particle_list_sig[i,2, 0] = ptdipho
            particle_list_sig[i,2, 1] = etadipho
            particle_list_sig[i,2, 2] = phidipho
            particle_list_sig[i,2, 3] = 0 # isLep
            particle_list_sig[i,2, 4] = 1 if ptdipho != 0 else 0 # isDiPho
            particle_list_sig[i,2, 5] = 0 # isMET

            # 3: MET
            particle_list_sig[i,3, 0] = ptMET
            particle_list_sig[i,3, 1] = etaMET
            particle_list_sig[i,3, 2] = phiMET
            particle_list_sig[i,3, 3] = 0 #isLep
            particle_list_sig[i,3, 4] = 0 # isDiPho
            particle_list_sig[i,3, 5] = 1 if ptMET != 0 else 0 # isMET
        
            # Sort by descending pT. 
            # This was implemented when standardization was done before sorting. Thus zero entry needs to be excluded
            # Redesigned the code with standardization done after sorting. Same code still works.
            nonzero_indices = np.nonzero(particle_list_sig[i,:,0])[0]
            sorted_indices = particle_list_sig[i,nonzero_indices,0].argsort()[::-1] # sort by first column, which is the pT
            global_sorted_indices = nonzero_indices[sorted_indices]
            sorted_particle_list[i,:len(nonzero_indices),:] = particle_list_sig[i,global_sorted_indices,:]
            
        return sorted_particle_list

    sig_train_list = to_p_list(sig_train_frame)
    sig_test_list = to_p_list(sig_test_frame)
    bkg_train_list = to_p_list(bkg_train_frame)
    bkg_test_list = to_p_list(bkg_test_frame)

    # Standardize the particle list
    x_sample = bkg_train_list[:,:,:3] # don't standardize boolean flags
    # Flatten out
    x_flat = x_sample.reshape((x_sample.shape[0]*x_sample.shape[1], x_sample.shape[2]))
    # Masked out zero
    zero_entries = (x_flat == 0)
    masked_x_sample = np.ma.array(x_flat, mask=zero_entries)
    x_list_mean = masked_x_sample.mean(axis=0)
    x_list_std = masked_x_sample.std(axis=0)
    del x_sample, x_flat, zero_entries, masked_x_sample # release the memory

    def standardize_p_list(inputs):
        to_norm = inputs[:,:,:3]
        zero_entries = (to_norm == 0)
        masked_to_norm = np.ma.array(to_norm, mask=zero_entries)
        normed_x = (masked_to_norm - x_list_mean)/x_list_std
        return np.concatenate((normed_x.filled(0), inputs[:,:,3:]), axis=2)
        
    normed_sig_list = standardize_p_list(sig_train_list)
    normed_sig_test_list = standardize_p_list(sig_test_list)
    normed_bkg_list = standardize_p_list(bkg_train_list)
    normed_bkg_test_list = standardize_p_list(bkg_test_list)

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

    # downsampling
    background_list = normed_bkg_list[:len(normed_sig_list)] 
    background_test_list = normed_bkg_test_list[:len(normed_sig_test_list)]
    background_hlf = normed_bkg_hlf[:len(normed_sig_hlf)]
    background_test_hlf = normed_bkg_test_hlf[:len(normed_sig_test_hlf)]
    background_train_aux = bkg_aux_train_frame.loc[:len(sig_aux_train_frame)]
    background_test_aux = bkg_aux_test_frame.loc[:len(sig_aux_test_frame)]

    sig_label = np.ones(len(normed_sig_hlf))
    bkg_label = np.zeros(len(background_hlf))
    sig_test_label = np.ones(len(normed_sig_test_hlf))
    bkg_test_label = np.zeros(len(background_test_hlf))

    # Build train data arrays
    data_list = np.concatenate((normed_sig_list, background_list))
    data_hlf = np.concatenate((normed_sig_hlf, background_hlf))
    label = np.concatenate((sig_label, bkg_label))
    # Shuffle train arrays
    p = rng.permutation(len(data_list))
    data_list, data_hlf, label = data_list[p], data_hlf[p], label[p]
    # Build and shuffle aux df
    data_aux = pd.concat([sig_aux_train_frame, background_train_aux], ignore_index=True)
    data_aux = data_aux.reindex(p)
    print("Data list: {}".format(data_list.shape))
    print("Data HLF: {}".format(data_hlf.shape))

    # Build test data arrays
    data_list_test = np.concatenate((normed_sig_test_list, background_test_list))
    data_hlf_test = np.concatenate((normed_sig_test_hlf, background_test_hlf))
    label_test = np.concatenate((sig_test_label, bkg_test_label))
    # Shuffle test arrays
    p_test = rng.permutation(len(data_list_test))
    data_list_test, data_hlf_test, label_test = data_list_test[p_test], data_hlf_test[p_test], label_test[p_test]
    # Build and shuffle aux df
    data_test_aux = pd.concat([sig_aux_test_frame, background_test_aux], ignore_index=True)
    data_test_aux = data_test_aux.reindex(p_test)
    print("Data list test: {}".format(data_list_test.shape))
    print("Data HLF test: {}".format(data_hlf_test.shape))

    if return_pre_std:
        return (
            sig_train_frame, sig_test_frame, 
            bkg_train_frame, bkg_test_frame, 
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