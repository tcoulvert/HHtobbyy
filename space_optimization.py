#Stdlib packages
import json

# Common Py packages
import numpy as np

# ML packages
import torch
import torch.nn as nn
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from torch.utils.data import DataLoader

# Module packages
from AMSGrad import AMSGrad
from InclusiveNetwork import InclusiveNetwork
from ParticleHLF import ParticleHLF
from train import train

def optimize_hyperparams_RR(
        data_list_dict, data_hlf_dict, label_dict, weight_dict, config_filename, NUM_EPOCHS=100, SEED=21
    ):
    space  = [
        Integer(1, 10, name='hidden_layers'),
        Integer(10, 500, name='initial_nodes'),
        Real(0.01,0.9,name='dropout'),
        Integer(1, 10, name='gru_layers'),
        Integer(10, 500, name='gru_size'),
        Real(0.01, 0.9, name='dropout_g'),
        Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
        Integer(4000, 4001, name='batch_size'),
        Real(10**-5, 10**-4, "log-uniform", name='L2_reg')
    ]
    # L1 reg: https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch

    @use_named_args(space)
    def objective(**X):
        print("New configuration: {}".format(X))
        fom = []
        for fold_idx in range(len(data_list_dict)):
            model = InclusiveNetwork(
                int(X['hidden_layers']), 
                int(X['initial_nodes']), 
                float(X['dropout']), 
                int(X['gru_layers']), 
                int(X['gru_size']), 
                float(X['dropout_g']),
                dnn_input=np.shape(data_hlf_dict['fold_0'])[-1],
                rnn_input=np.shape(data_list_dict['fold_0'])[-1],
            ).cuda()
            optimizer = AMSGrad(model.parameters(), lr=float(X['learning_rate']), weight_decay=float(X['L2_reg']))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            (
                train_data_list, val_data_list,
                train_data_hlf, val_data_hlf,
                train_label, val_label,
                train_weight, val_weight
            ) = train_test_split(
                data_list_dict[f'fold_{fold_idx}'], data_hlf_dict[f'fold_{fold_idx}'], label_dict[f'fold_{fold_idx}'], weight_dict[f'fold_{fold_idx}'],
                test_size=0.2, random_state=SEED
            )
            train_loader = DataLoader(
                ParticleHLF(train_data_list, train_data_hlf, train_label, train_weight), 
                batch_size=int(X['batch_size']), shuffle=True
            )
            val_loader = DataLoader(
                ParticleHLF(val_data_list, val_data_hlf, val_label, val_weight), 
                batch_size=int(X['batch_size']), shuffle=True
            )
            data_loader = {"training": train_loader, "validation": val_loader}

            best_acc, train_losses, val_losses = train(
                NUM_EPOCHS, model, optimizer, scheduler, 
                data_loader=data_loader
            )

            fom.append(best_acc)
        Y = np.mean(np.asarray([acc.cpu() for acc in fom]))
        print("Average best_acc across k-fold: {}".format(Y))
        return -Y

    res_gp = gp_minimize(objective, space, n_calls=30, random_state=0)

    print("Best parameters: {}".format(res_gp.x))
    best_hidden_layers = int(res_gp.x[0])
    best_initial_nodes = int(res_gp.x[1])
    best_dropout = float(res_gp.x[2])
    best_gru_layers = int(res_gp.x[3])
    best_gru_size = int(res_gp.x[4])
    best_dropout_g = float(res_gp.x[5])
    best_learning_rate = float(res_gp.x[6])
    best_batch_size = int(res_gp.x[7])
    best_L2_reg = float(res_gp.x[8])

    best_conf = {
        "hidden_layers": best_hidden_layers,
        "initial_nodes": best_initial_nodes,
        "dropout": best_dropout,
        "gru_layers": best_gru_layers,
        "gru_size": best_gru_size,
        "dropout_g": best_dropout_g,
        "learning_rate": best_learning_rate,
        "batch_size": best_batch_size,
        "L2_reg": best_L2_reg
    }
    with open(config_filename, 'w') as config:
        json.dump(best_conf, config)
        # print("Save best configuration to {}".format(config_filename))
    return best_conf


def optimize_hyperparams(
        skf, data_list, data_hlf, label, weight, config_filename, epochs=100
    ):
    space  = [
        Integer(1, 3, name='hidden_layers'),
        Integer(10, 500, name='initial_nodes'),
        Real(0.01,0.9,name='dropout'),
        Integer(2, 3, name='gru_layers'),
        Integer(10, 500, name='gru_size'),
        Real(0.01,0.9,name='dropout_g'),
        Real(10**-5, 10**-1, "log-uniform", name='learning_rate'),
        Integer(4000,4001,name='batch_size'),
        # Integer(32,512,name='batch_size'),
        Real(10**-5, 10**-4, "log-uniform", name='L2_reg')
    ]
    # L1 reg: https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
    # batch_size = 4000

    @use_named_args(space)
    def objective(**X):
        print("New configuration: {}".format(X))
        fom = []
        for train_index, test_index in skf.split(data_hlf, label):
            train_loader = DataLoader(
                ParticleHLF(data_list[train_index], data_hlf[train_index], label[train_index], weight[train_index]), 
                batch_size=int(X['batch_size']), 
                shuffle=True
            )
            val_loader = DataLoader(
                ParticleHLF(data_list[test_index], data_hlf[test_index], label[test_index], weight[train_index]), 
                batch_size=int(X['batch_size']), 
                shuffle=True
            )
            data_loader = {"training": train_loader, "validation": val_loader} 
            # print(train_loader)

            model = InclusiveNetwork(
                int(X['hidden_layers']), 
                int(X['initial_nodes']), 
                float(X['dropout']), 
                int(X['gru_layers']), 
                int(X['gru_size']), 
                float(X['dropout_g']),
                dnn_input=np.shape(data_hlf)[-1],
                rnn_input=np.shape(data_list)[-1],
            ).cuda()
            # model = InclusiveNetwork(X['hidden_layers'], X['initial_nodes'], X['dropout'], X['gru_layers'], X['gru_size'], X['dropout_g'])

            optimizer = AMSGrad(model.parameters(), lr=X['learning_rate'], weight_decay=X['L2_reg'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor=0.5,patience=4)
            best_acc, train_losses, val_losses = train(
                epochs, model, optimizer, scheduler,
                data_loader=data_loader
            )
            fom.append(best_acc)
        Y = np.mean(np.asarray([acc.cpu() for acc in fom]))
        print("Average best_acc across k-fold: {}".format(Y))
        return -Y

    res_gp = gp_minimize(objective, space, n_calls=30, random_state=0)

    print("Best parameters: {}".format(res_gp.x))
    best_hidden_layers = int(res_gp.x[0])
    best_initial_nodes = int(res_gp.x[1])
    best_dropout = float(res_gp.x[2])
    best_gru_layers = int(res_gp.x[3])
    best_gru_size = int(res_gp.x[4])
    best_dropout_g = float(res_gp.x[5])
    best_learning_rate = float(res_gp.x[6])
    best_batch_size = int(res_gp.x[7])
    best_L2_reg = float(res_gp.x[8])

    best_conf = {
        "hidden_layers": best_hidden_layers,
        "initial_nodes": best_initial_nodes,
        "dropout": best_dropout,
        "gru_layers": best_gru_layers,
        "gru_size": best_gru_size,
        "dropout_g": best_dropout_g,
        "learning_rate": best_learning_rate,
        "batch_size": best_batch_size,
        "L2_reg": best_L2_reg
    }
    with open(config_filename, 'w') as config:
        json.dump(best_conf, config)
        # print("Save best configuration to {}".format(config_filename))
    return best_conf
