# Stdlib packages #
import copy
import glob
import json
import os
import re
from pathlib import Path

# Common Py packages #
import numpy as np
import pandas as pd

# HEP Packages #
import awkward as ak

# ML packages #
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import roc_curve, auc
from sklearn.metrics import auc
from torch.utils.data import DataLoader, Dataset

# Prefix for parquet files #
PARQUET_FILEPREFIX = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v1"

# NN Dataset class #
class ParticleHLF(Dataset):
    def __init__(self, data_particles, data_hlf, data_y, data_weights):
        self.len = data_y.shape[0]
        self.data_particles = torch.from_numpy(data_particles).float()
        self.data_hlf = torch.from_numpy(data_hlf).float()
        self.data_y = torch.from_numpy(data_y).long()
        self.data_weight = torch.from_numpy(data_weights).float()
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.data_particles[idx], self.data_hlf[idx], self.data_y[idx], self.data_weight[idx])

# NN architecture class #
class InclusiveNetwork(nn.Module):
    def __init__(
            self, num_hiddens=2, initial_node=500, dropout=0.5, gru_layers=2, gru_size=50, 
            dropout_g=0.1, rnn_input=6, dnn_input=21
        ):
        super(InclusiveNetwork, self).__init__()
        self.dropout = dropout
        self.dropout_g = dropout_g
        self.hiddens = nn.ModuleList()
        nodes = [initial_node]
        for i in range(num_hiddens):
            nodes.append(int(nodes[i]/2))
            self.hiddens.append(nn.Linear(nodes[i],nodes[i+1]))
        self.gru = nn.GRU(input_size=rnn_input, hidden_size=gru_size, num_layers=gru_layers, batch_first=True, dropout=self.dropout_g)
        self.merge = nn.Linear(dnn_input+gru_size,initial_node)
        self.out = nn.Linear(nodes[-1],2)

    def forward(self, particles, hlf):
        _, hgru = self.gru(particles)
        hgru = hgru[-1] # Get the last hidden layer
        x = torch.cat((hlf,hgru), dim=1)
        x = F.dropout(self.merge(x), training=self.training, p=self.dropout)
        for i in range(len(self.hiddens)):
            x = F.relu(self.hiddens[i](x))
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.out(x)
        return F.log_softmax(x, dim=1)
    
# Process the data in the way the model expects #
def process_data(
    sample_ak, seed=None,
):
    hlf_list = [
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
        'n_leptons', 
        'lead_bjet_pt', 'lead_bjet_eta', 'lead_bjet_phi',
        'sublead_bjet_pt', 'sublead_bjet_eta', 'sublead_bjet_phi',
    ]
    hlf_list.sort()
    sample_pd = pd.DataFrame({
        field: ak.to_numpy(sample_ak[field], allow_missing=False) for field in hlf_list
    })
    aux_list = [
        'event'
    ]
    aux_list.sort()
    aux_pd = pd.DataFrame({
        field: ak.to_numpy(sample_ak[field], allow_missing=False) for field in aux_list
    })

    # Randomly shuffle DFs and split into train and test samples #
    rng = np.random.default_rng(seed=seed)
    sample_idx = rng.permutation(sample_pd.index)
    sample_pd = sample_pd.reindex(sample_idx)
    aux_pd = aux_pd.reindex(sample_idx)

    def train_test_split_df(df, aux_df, dataset_num=0):
        # Train/Val events are those with event#s ≠ dataset_num, test events have even event#s = dataset_num
        train_df = df.loc[(aux_df['event'] % 5).ne(dataset_num)].reset_index(drop=True)
        test_df = df.loc[(aux_df['event'] % 5).eq(dataset_num)].reset_index(drop=True)

        train_aux_df = aux_df.loc[(aux_df['event'] % 5).ne(dataset_num)].reset_index(drop=True)
        test_aux_df = aux_df.loc[(aux_df['event'] % 5).eq(dataset_num)].reset_index(drop=True)

        return train_df, test_df, train_aux_df, test_aux_df
    
    for fold in range(5):
        (
            train_df, test_df, 
            train_aux_df, test_aux_df  # Due to the way the training is done, only 'test' events should be evaluated
        ) = train_test_split_df(sample_pd, aux_pd, dataset_num=fold)


        # Perform the standardization #
        with open('standardization.json', 'r') as f:
            std_dict = json.load(f)

        for col_idx, col in enumerate(std_dict['standardized_variables']):
            if std_dict['standardized_logs']:
                test_df[col] = np.where(test_df[col] > 0, np.log(test_df[col]), test_df[col])
            test_df[col] = (test_df[col] - std_dict['standardized_mean'][col_idx]) / std_dict['standardized_stddev'][col_idx]

        no_standardize = {
            'puppiMET_eta', 'puppiMET_phi', # MET variables
            'DeltaPhi_j1MET', 'DeltaPhi_j2MET', # jet-MET variables
            'lepton1_eta', 'lepton2_eta', 'eta', # lepton and diphoton eta
            'lepton1_phi', 'lepton2_phi', 'phi', # lepton and diphoton phi
            'CosThetaStar_CS','CosThetaStar_jj',
            'lead_bjet_eta', 'lead_bjet_phi',
            'sublead_bjet_eta', 'sublead_bjet_phi',
        }
        def apply_log(df):
            log_fields = {
                'puppiMET_sumEt', 
                'puppiMET_pt', # MET variables
                'chi_t0', 'chi_t1', # jet variables
                'lepton1_pt' ,'lepton2_pt', 'pt', # lepton and diphoton pt
                'lead_bjet_pt', 'sublead_bjet_pt', # bjet pts
                'dijet_mass', # mass of b-dijet (resonance for H->bb)
            }
            for field in (log_fields & set(hlf_list)) - no_standardize:
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
        with open(output_dirpath + 'standardization.json', 'w') as f:
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
                
            #     for i in range(n_particle_fields, len(extra_RNN_vars)):
            #         particle_list_sig[:, var_idx, i] = np.zeros_like(data_frame[var_name+'pt'].to_numpy())
            
            # for var_idx, var_name in enumerate(extra_RNN_vars, start=4):
            #     particle_list_sig[:, var_idx, 0] = np.where(data_frame[var_name].to_numpy() != train_pad[col_idx_dict[var_name]], data_frame[var_name].to_numpy(), 0)
            #     particle_list_sig[:, var_idx, 1:] = np.zeros_like(particle_list_sig[:, 0, 1:])
            #     particle_list_sig[:, var_idx, 2+var_idx] = np.ones_like(particle_list_sig[:, var_idx, 0])

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

# Evaluate the data over the trained model #
def fill_array(array_to_fill, value, index, batch_size):
    array_to_fill[index*batch_size:min((index+1)*batch_size, array_to_fill.shape[0])] = value  

def evaluate(
        p_list, hlf, label, weight,
        best_conf,
    ):
    model = InclusiveNetwork(
        best_conf['hidden_layers'], best_conf['initial_nodes'], best_conf['dropout'], 
        best_conf['gru_layers'], best_conf['gru_size'], best_conf['dropout_g'], 
        dnn_input=len(hlf[0] if not dict_lists or only_fold_idx is not None else hlf['fold_0'][0]), rnn_input=len(p_list[0, 0, :] if not dict_lists or only_fold_idx is not None else p_list['fold_0'][0, 0, :]),
    ).cuda()

    fprs = []
    base_tpr = np.linspace(0, 1, 5000)
    thresholds = []
    best_batch_size = best_conf['batch_size']
    
    all_preds, all_labels = [], []

    for fold_idx in [only_fold_idx] if only_fold_idx is not None else (range(skf.get_n_splits() if not dict_lists else len(p_list))):
        # if only_fold_idx is not None and fold_idx != only_fold_idx:
        #     continue
        model.load_state_dict(torch.load(OUTPUT_DIRPATH + f'{CURRENT_TIME}_ReallyTopclassStyle_{fold_idx}.torch'))
        model.eval()
        if not dict_lists:
            all_pred = np.zeros(shape=(len(hlf),2))
            all_label = np.zeros(shape=(len(hlf)))
            val_loader = DataLoader(
                ParticleHLF(p_list, hlf, label, weight), 
                batch_size=best_conf['batch_size'],
                shuffle=False
            )
        else:
            all_pred = np.zeros(shape=(len(hlf[f"fold_{fold_idx}"]),2))
            all_label = np.zeros(shape=(len(hlf[f"fold_{fold_idx}"])))
            val_loader = DataLoader(
                ParticleHLF(p_list[f"fold_{fold_idx}"], hlf[f"fold_{fold_idx}"], label[f"fold_{fold_idx}"], weight[f"fold_{fold_idx}"]), 
                batch_size=best_conf['batch_size'],
                shuffle=False
            )
        with torch.no_grad():
            for batch_idx, (particles_data, hlf_data, y_data, weight_data) in enumerate(val_loader):
                
                # print(f"val_loader: {batch_idx}")
                particles_data = particles_data.numpy()
                arr = np.sum(particles_data!=0, axis=1)[:,0] # the number of particles in the whole batch
                arr = [1 if x==0 else x for x in arr]
                arr = np.array(arr)
                sorted_indices_la = np.argsort(-arr)
                particles_data = torch.from_numpy(particles_data[sorted_indices_la]).float()
                hlf_data = hlf_data[sorted_indices_la]
                particles_data = Variable(particles_data).cuda()
                hlf_data = Variable(hlf_data).cuda()
                # particles_data = Variable(particles_data)
                # hlf_data = Variable(hlf_data)
                t_seq_length = [arr[i] for i in sorted_indices_la]
                particles_data = torch.nn.utils.rnn.pack_padded_sequence(particles_data, t_seq_length, batch_first=True)

                outputs = model(particles_data, hlf_data)

                # Unsort the predictions (to match the original data order)
                # https://stackoverflow.com/questions/34159608/how-to-unsort-a-np-array-given-the-argsort
                b = np.argsort(sorted_indices_la)
                unsorted_pred = outputs[b].data.cpu().numpy()

                fill_array(all_pred, unsorted_pred, batch_idx, best_batch_size)
                fill_array(all_label, y_data.numpy(), batch_idx, best_batch_size)

        fpr, tpr, threshold = roc_curve(all_label, np.exp(all_pred)[:,1])
        # print(threshold)

        fpr = np.interp(base_tpr, tpr, fpr)
        threshold = np.interp(base_tpr, tpr, threshold)
        fpr[0] = 0.0
        fprs.append(fpr)
        thresholds.append(threshold)
        all_preds.append(copy.deepcopy(all_pred.tolist()))
        all_labels.append(copy.deepcopy(all_label.tolist()))

    thresholds = np.array(thresholds)
    mean_thresholds = thresholds.mean(axis=0)

    fprs = np.array(fprs)
    mean_fprs = fprs.mean(axis=0)
    std_fprs = fprs.std(axis=0)
    fprs_right = np.minimum(mean_fprs + std_fprs, 1)
    fprs_left = np.maximum(mean_fprs - std_fprs, 0)

    mean_area = auc(mean_fprs, base_tpr)
    areas = [float(auc(fprs[fpr_fold], base_tpr)) for fpr_fold in range(np.shape(fprs)[0])]

    if val_losses_arr is None and train_losses_arr is None:
        with open(OUTPUT_DIRPATH + f'{CURRENT_TIME}_IN_perf.json', 'r') as f:
            old_IN_perf = json.load(f)
        IN_perf = {
            'train_losses_arr': old_IN_perf['train_losses_arr'],
            'val_losses_arr': old_IN_perf['val_losses_arr'],
            'fprs': fprs.tolist(),
            'mean_fprs': mean_fprs.tolist(),
            'std_fprs': std_fprs.tolist(),
            'fprs_right': fprs_right.tolist(),
            'fprs_left': fprs_left.tolist(),
            'thresholds': thresholds.tolist(),
            'mean_thresholds': mean_thresholds.tolist(),
            'base_tpr': base_tpr.tolist(),
            'mean_area': float(mean_area),
            'all_areas': areas,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'mean_pred': np.mean(all_preds, axis=0).tolist(),
            'mean_label': np.mean(all_labels, axis=0).tolist(),
        }
    else:
        IN_perf = {
            'train_losses_arr': train_losses_arr,
            'val_losses_arr': val_losses_arr,
            'fprs': fprs.tolist(),
            'mean_fprs': mean_fprs.tolist(),
            'std_fprs': std_fprs.tolist(),
            'fprs_right': fprs_right.tolist(),
            'fprs_left': fprs_left.tolist(),
            'thresholds': thresholds.tolist(),
            'mean_thresholds': mean_thresholds.tolist(),
            'base_tpr': base_tpr.tolist(),
            'mean_area': float(mean_area),
            'all_areas': areas,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'mean_pred': np.mean(all_preds, axis=0).tolist(),
            'mean_label': np.mean(all_labels, axis=0).tolist(),
        }
    if save:
        with open(OUTPUT_DIRPATH + f'{CURRENT_TIME}_IN_perf.json', 'w') as f:
            json.dump(IN_perf, f)
        with h5py.File(OUTPUT_DIRPATH + f"{CURRENT_TIME}_ReallyInclusive_ROC.h5","w") as out:
            out['FPR'] = mean_fprs
            out['dFPR'] = std_fprs
            out['TPR'] = base_tpr
            out['Thresholds'] = mean_thresholds

    return IN_perf

def main():
    parquet_filepath_list = glob.glob(PARQUET_FILEPREFIX+'/**.parquet')

    for parquet_filepath in parquet_filepath_list:
        sample = ak.from_parquet(parquet_filepath)

    

    for data_era, dir_list in dir_lists.items():
        for dir_name in dir_list:
            if dir_name not in {'ttHToGG', 'GluGluToHH'}:
                continue
            for sample_type in ['nominal']: # Eventually change to os.listdir(LPC_FILEPREFIX+'/'+data_era+'/'+dir_name)
                # Load all the parquets of a single sample into an ak array
                
                add_ttH_vars(sample)
        
                if re.match('Data', dir_name) is None:
                    # Compute the sum of genWeights for proper MC rescaling.
                    sample['sumGenWeights'] = sum(
                        float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in glob.glob(
                            LPC_FILEPREFIX+'/'+data_era+'/'+dir_name+'/'+sample_type+'/*'
                        )
                    )
        
                    # Store luminostity computed from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
                    #   and summing over lumis of the same type (e.g. all 22EE era lumis summed).
                    sample['luminosity'] = luminosities[data_era]
            
                    # If the process has a defined cross section, use defined xs otherwise use 1e-3 [fb] for now.
                    sample['cross_section'] = cross_sections[dir_name]
        
                    # Define eventWeight array for hist plotting.
                    # print('========================')
                    # abs_genWeight = ak.where(sample['genWeight'] < 0, -sample['genWeight'], sample['genWeight'])
                    # sum_of_abs_genWeight = ak.sum(ak.where(sample['genWeight'] < 0, -1, 1), axis=0)
                    # sample['eventWeight'] = ak.where(sample['genWeight'] < 0, -1, 1) * (sample['luminosity'] * sample['cross_section'] / sum_of_abs_genWeight)
                    sample['eventWeight'] = sample['genWeight'] * (sample['luminosity'] * sample['cross_section'] / sample['sumGenWeights'])
        
                destdir = LPC_FILEPREFIX+'/'+data_era+'_merged_v2/'+dir_name+'/'+sample_type+'/'
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                merged_parquet = ak.to_parquet(sample, destdir+dir_name+'_'+sample_type+'.parquet')
                
                del sample
                print('======================== \n', dir_name)
                run_samples['run_samples_list'].append(dir_name)
                with open(LPC_FILEPREFIX+'/'+data_era+'/completed_samples.json', 'w') as f:
                     json.dump(run_samples, f)


if __name__ == '__main__':
    main()