# Stdlib packages #
import argparse
import copy
import glob
import json
import os
import re
import warnings

# Common Py packages #
import numpy as np
import pandas as pd

# HEP Packages #
import awkward as ak

# ML packages #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

PARQUET_FILEPREFIX = ""  # Prefix for parquet files
MODEL_FILEPREFIX = ""  # Prefix for model files
FILL_VALUE = -999  # Fill value for bad data in parquet files
SEED = None  # Seed for rng

# NN Dataset class #
class ParticleHLF(Dataset):
    def __init__(self, data_particles, data_hlf):
        self.len = data_particles.shape[0]
        self.data_particles = torch.from_numpy(data_particles).float()
        self.data_hlf = torch.from_numpy(data_hlf).float()
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.data_particles[idx], self.data_hlf[idx])

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
    sample_ak, std_dict_filepath_list, nFolds=5
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
    }, dtype=np.float64)
    aux_list = [
        'event'
    ]
    aux_list.sort()
    aux_pd = pd.DataFrame({
        field: ak.to_numpy(sample_ak[field], allow_missing=False) for field in aux_list
    })

    # Randomly shuffle DFs and split into train and test samples #
    rng = np.random.default_rng(seed=SEED)
    sample_idx = rng.permutation(sample_pd.index)
    sample_pd = sample_pd.reindex(sample_idx)
    aux_pd = aux_pd.reindex(sample_idx)

    def train_test_split_df(df, aux_df, dataset_num=0):
        # Train/Val events are those with event#s ≠ dataset_num, test events have even event#s = dataset_num
        train_df = df.loc[(aux_df['event'] % nFolds).ne(dataset_num)].reset_index(drop=True)
        test_df = df.loc[(aux_df['event'] % nFolds).eq(dataset_num)].reset_index(drop=True)

        train_aux_df = aux_df.loc[(aux_df['event'] % nFolds).ne(dataset_num)].reset_index(drop=True)
        test_aux_df = aux_df.loc[(aux_df['event'] % nFolds).eq(dataset_num)].reset_index(drop=True)

        return train_df, test_df, train_aux_df, test_aux_df
    
    # Make the p-list for the RNN #
    def to_p_list(data_frame):
        # Inputs: Pandas data frame
        # Outputs: Numpy array of dimension (Event, Particle, Attributes)
        
        # 6 particles: lead/sublead lepton, MET, diphoton, lead/sublead bjet
        # 7 fields per particle: pt, eta, phi, 4D one-hot encoding of particle type
        particle_list_sig = np.zeros(shape=(len(data_frame), 6, 7))

        var_names = ['lepton1', 'lepton2', '', 'puppiMET', 'lead_bjet', 'sublead_bjet']
        var_one_hots = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
        data_types = {0: 'pt', 1: 'eta', 2: 'phi'}

        for var_idx, var_name in enumerate(var_names):
            if var_name != '':
                var_name = var_name + '_'

            for local_idx, data_type in data_types.items():
                particle_list_sig[:, var_idx, local_idx] = np.where(
                    data_frame[var_name+data_type].to_numpy() != std_dict['standardized_unphysical_values'][std_dict['standardized_variables'].index(var_name+data_type)], 
                    data_frame[var_name+data_type].to_numpy(), 0
                )
            
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
    
    for fold in range(nFolds):
        (
            train_df, test_df, 
            train_aux_df, test_aux_df  # Due to the way the training is done, only 'test' events should be evaluated
        ) = train_test_split_df(sample_pd, aux_pd, dataset_num=fold)


        # Perform the standardization #
        with open(std_dict_filepath_list[fold], 'r') as f:
            std_dict = json.load(f)
        for col_idx, col in enumerate(std_dict['standardized_variables']):
            if std_dict['standardized_logs'][col_idx]:
                positive_mask = (test_df.loc[:, col].to_numpy() > 0)
                test_df.loc[positive_mask, col] = np.log(test_df.loc[positive_mask, col].to_numpy())
            
            unphysical_mask_arr = (test_df.loc[:, col].to_numpy() == FILL_VALUE)

            test_df.loc[:, col] = (
                test_df.loc[:, col].to_numpy() - std_dict['standardized_mean'][col_idx]
            ) / std_dict['standardized_stddev'][col_idx]

            test_df.loc[unphysical_mask_arr, col] = std_dict['standardized_unphysical_values'][col_idx]
        
        normed_test_list = to_p_list(test_df)

        input_hlf_vars_max = [
            'puppiMET_sumEt',
            'n_jets','chi_t0', 'chi_t1',
            'CosThetaStar_CS','CosThetaStar_jj', 
            'DeltaR_jg_min',
            'DeltaPhi_j1MET','DeltaPhi_j2MET',
            'leadBjet_leadLepton', 'leadBjet_subleadLepton', 'subleadBjet_leadLepton', 'subleadBjet_subleadLepton', 
            'dijet_mass',
            'n_leptons'
        ]
        input_hlf_vars = []
        for var in input_hlf_vars_max:
            if var in set(hlf_list):
                input_hlf_vars.append(var)
        input_hlf_vars.sort()

        normed_test_hlf = test_df[input_hlf_vars].values

        # Build and shuffle data arrays
        p = rng.permutation(len(normed_test_list))
        data_list, data_hlf = normed_test_list[p], normed_test_hlf[p]
        data_aux = (test_aux_df.reindex(p)).reset_index(drop=True)

        if fold == 0:
            full_data_list, full_data_hlf, full_data_aux = (
                {f'fold_{0}': copy.deepcopy(data_list)}, 
                {f'fold_{0}': copy.deepcopy(data_hlf)},
                {f'fold_{0}': copy.deepcopy(data_aux)}
            )
        else:
            full_data_list[f'fold_{fold}'], full_data_hlf[f'fold_{fold}'], full_data_aux[f'fold_{fold}'] = (
                copy.deepcopy(data_list), copy.deepcopy(data_hlf), copy.deepcopy(data_aux)
            )
            if fold == 4:
                return full_data_list, full_data_hlf, full_data_aux

# Evaluate the data over the trained model #
def fill_array(array_to_fill, value, index, batch_size):
    array_to_fill[index*batch_size:min((index+1)*batch_size, array_to_fill.shape[0])] = value  

def evaluate(
    p_list, hlf, best_conf, model_filepath_list
):
    model = InclusiveNetwork(
        best_conf['hidden_layers'], best_conf['initial_nodes'], best_conf['dropout'], 
        best_conf['gru_layers'], best_conf['gru_size'], best_conf['dropout_g'], 
        dnn_input=len(hlf['fold_0'][0]), rnn_input=len(p_list['fold_0'][0, 0, :]),
    )

    best_batch_size = best_conf['batch_size']
    
    all_preds = []

    for fold_idx in range(len(p_list)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.load_state_dict(torch.load(model_filepath_list[fold_idx]))
        model.eval()
        all_pred = np.zeros(shape=(len(hlf[f"fold_{fold_idx}"]),2))
        eval_loader = DataLoader(
            ParticleHLF(p_list[f"fold_{fold_idx}"], hlf[f"fold_{fold_idx}"]), 
            batch_size=best_conf['batch_size'],
            shuffle=False
        )
        with torch.no_grad():
            for batch_idx, (particles_data, hlf_data) in enumerate(eval_loader):

                particles_data = particles_data.numpy()
                arr = np.sum(particles_data!=0, axis=1)[:,0] # the number of particles in the whole batch
                arr = [1 if x==0 else x for x in arr]
                arr = np.array(arr)
                sorted_indices_la = np.argsort(-arr)
                particles_data = torch.from_numpy(particles_data[sorted_indices_la]).float()
                hlf_data = hlf_data[sorted_indices_la]
                particles_data = Variable(particles_data)
                hlf_data = Variable(hlf_data)
                t_seq_length = [arr[i] for i in sorted_indices_la]
                particles_data = torch.nn.utils.rnn.pack_padded_sequence(particles_data, t_seq_length, batch_first=True)

                outputs = model(particles_data, hlf_data)

                # Unsort the predictions (to match the original data order)
                # https://stackoverflow.com/questions/34159608/how-to-unsort-a-np-array-given-the-argsort
                b = np.argsort(sorted_indices_la)
                unsorted_pred = outputs[b].data.numpy()

                fill_array(all_pred, unsorted_pred, batch_idx, best_batch_size)

        all_preds.append(copy.deepcopy(all_pred))

    return all_preds

# Sorts the predictions to map the output to the correct event
def sorted_preds(preds, data_aux, sample):
    flat_preds = np.concatenate([np.exp(preds[fold_idx])[:, 1] for fold_idx in range(len(data_aux))])
    preds_sort = np.argsort(
        np.concatenate([data_aux[f"fold_{fold_idx}"].loc[:, 'event'].to_numpy() for fold_idx in range(len(data_aux))])
    )

    sample_sort = np.argsort(np.argsort(
        ak.to_numpy(sample['event'], allow_missing=False)
    ))

    return flat_preds[preds_sort][sample_sort]

# Runs the script to add ttH-killer preds to the samples #
def main(output_dirpath=None):
    # list of parquet filepaths
    parquet_filepath_list = glob.glob(os.path.join(PARQUET_FILEPREFIX, '**/*.parquet'), recursive=True)

    # list of model filepaths (N models for N folds)
    model_filepath_list = glob.glob(os.path.join(MODEL_FILEPREFIX, '*.torch'))
    model_filepath_list.sort()

    # filepath for model config file
    with open(glob.glob(os.path.join(MODEL_FILEPREFIX, '*config.json'))[0], 'r') as f:
        best_conf = json.load(f)

    # list of standardization filepaths (N stds for N models)
    std_dict_filepath_list = glob.glob(os.path.join(MODEL_FILEPREFIX, '*standardization.json'))
    std_dict_filepath_list.sort()

    # Performs the eval for each parquet and saves out the new parquet with the added ttH score field
    for parquet_filepath in parquet_filepath_list:
        sample = ak.from_parquet(parquet_filepath)

        data_list, data_hlf, data_aux = process_data(sample, std_dict_filepath_list, nFolds=len(model_filepath_list))

        preds = evaluate(data_list, data_hlf, best_conf, model_filepath_list)

        sample['ttH_killer_preds'] = sorted_preds(preds, data_aux, sample)

        dest_filepath = parquet_filepath[:parquet_filepath.rfind('.')] + '_ttH_killer_preds' + parquet_filepath[parquet_filepath.rfind('.'):]
        if output_dirpath is not None:
            substr = dest_filepath[len(PARQUET_FILEPREFIX)+1:]
            dest_filepath = os.path.join(output_dirpath, dest_filepath[len(PARQUET_FILEPREFIX)+1:])
            if not os.path.exists(dest_filepath[:dest_filepath.rfind('/')]):
                os.makedirs(dest_filepath[:dest_filepath.rfind('/')])
        merged_parquet = ak.to_parquet(sample, dest_filepath)
        
        del sample
        print('======================== \n', dest_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process the HiggsDNA HHbbgg parquets to add the ttH-killer predictions.'
    )
    parser.add_argument('--parquet-fileprefix', action='store', required=True,
        help='The full path to the file prefix where the parquet files are located.'
    )
    parser.add_argument('--model-fileprefix', action='store', required=True,
        help='The full path to the file prefix where the trained ttH-Killer model files are located. This directory should contain only the model files, the standardization .json file, and the config .json file.'
    )
    parser.add_argument('--dump', dest='output_dirpath', action='store', default='none',
        help='Name of the output path in which the processed parquets will be stored. By default saves the parquets next to the old parquets with a name signifying they have been processed through the ttH-Killer.'
    )

    args = parser.parse_args()

    PARQUET_FILEPREFIX = args.parquet_fileprefix
    MODEL_FILEPREFIX = args.model_fileprefix

    main(output_dirpath=None if args.output_dirpath == 'none' else args.output_dirpath)