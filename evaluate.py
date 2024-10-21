# Stdlib packages
import copy
import json

# Common Py packages
import numpy as np
import h5py

# ML packages
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Module packages
from InclusiveNetwork import InclusiveNetwork
from ParticleHLF import ParticleHLF


def fill_array(array_to_fill, value, index, batch_size):
    array_to_fill[index*batch_size:min((index+1)*batch_size, array_to_fill.shape[0])] = value  

def evaluate(
        p_list, hlf, label, weight,
        OUTPUT_DIRPATH, CURRENT_TIME, skf, best_conf,
        train_losses_arr=None, val_losses_arr=None, save=False, only_fold_idx=None,
        dict_lists=False
        # aux_df=None
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
    # all_pred = np.zeros(shape=(len(hlf),2))
    # all_label = np.zeros(shape=(len(hlf)))

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