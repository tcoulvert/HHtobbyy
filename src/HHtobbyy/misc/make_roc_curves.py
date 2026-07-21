# Stdlib packages
import os

# Common py packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML packages
from sklearn.metrics import roc_curve
from scipy.integrate import trapezoid

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.models import map_model_to_Model

###########################################


# MLP
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_MLPv7_2026-07-15_08-13-50/dataset_config.json"
## MLP scikit val
## dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_MLPv7_2026-07-16_07-27-28/dataset_config.json"
# MLP 5fold
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_MLPv7F5_2026-07-18_14-10-32/dataset_config.json"
# BDT 5fold
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_BDTv7F5_2026-07-20_08-22-14/dataset_config.json"
# MulticlassBDT 5fold
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_MulticlassBDTv7F5_2026-07-19_22-16-35/dataset_config.json"
# MulticlassBDT 5fold LbTag in Train
dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_MulticlassBDTv7F5LbTag_2026-07-20_13-59-02/dataset_config.json"

# BDT Boosted
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_BDTv7_Boost_2026-07-15_23-03-07/dataset_config.json"
# BDT 5fold Boosted
# dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_BDTv7F5_Boost_2026-07-20_13-59-07/dataset_config.json"
## BDT Boosted scikit val
## dfdataset_config = "../HiggsDNA_parquet/DFDatasets/v3/16to25_BDTv7_Boost_2026-07-16_08-53-35/dataset_config.json"
dfdataset = DFDataset(dfdataset_config)
# model, model_config = "MLP", <filepath_to_model.json>
# model = map_model_to_Model(model)(dfdataset, model_config)
BASE_TPR = np.linspace(0, 1, 5000)

if '_Boost_' in dfdataset_config:
    plot_dir = os.path.join(os.getcwd(), "BDTv7F5_Boost_roc_curves")
    output_nodes = ["AUX_DBackground", "AUX_DSignal"]
    signals = {
        'ggFHH': (1, ['GluGluToHH_kl1p00_kt1p00_c20p00']),
        'VBFHH': (1, ['VBFToHH_cv1p00_c2v1p00_c31p00'])
    }
elif '_MulticlassBDTv7F5LbTag' in dfdataset_config:
    plot_dir = os.path.join(os.getcwd(), "MulticlassBDTv7LbTag_roc_curves")
    output_nodes = ["AUX_DggFHH", "AUX_DVBFHH", "AUX_DRes", "AUX_DnonRes"]
    signals = {
        'ggFHH': (0, ['GluGluToHH_kl1p00_kt1p00_c20p00']),
        'VBFHH': (1, ['VBFToHH_cv1p00_c2v1p00_c31p00'])
    }
elif '_MulticlassBDTv7' in dfdataset_config:
    plot_dir = os.path.join(os.getcwd(), "MulticlassBDTv7LbTag_roc_curves")
    output_nodes = ["AUX_DggFHH", "AUX_DttHbbH", "AUX_DVH", "AUX_DnonResggFHVBFH"]
    signals = {
        'ggFHH': (0, ['GluGluToHH_kl1p00_kt1p00_c20p00'])
    }
elif '_MLPv7' in dfdataset_config or '_BDTv7' in dfdataset_config:
    plot_dir = os.path.join(os.getcwd(), "BDTv7F5_roc_curves")
    output_nodes = ["AUX_DnonRes", "AUX_DRes", "AUX_DggFHH", "AUX_DVBFHH"]
    signals = {
        'ggFHH': (2, ['GluGluToHH_kl1p00_kt1p00_c20p00']),
        'VBFHH': (3, ['VBFToHH_cv1p00_c2v1p00_c31p00'])
    }
os.makedirs(plot_dir, exist_ok=True)
columns = ["AUX_eventWeight", "AUX_sample_name", "AUX_label1D"]+output_nodes
columns_map = {item: item for item in columns}

bkgs = {
    'All': ['GGJets', 'DDQCDGJets', 'TTGG', 'GluGluHToGG', 'VBFHToGG', 'ttHToGG', 'bbHToGG', 'VHToGG'],
    'nonRes': ['GGJets', 'DDQCDGJets', 'TTGG'],
    'Res': ['GluGluHToGG', 'VBFHToGG', 'ttHToGG', 'bbHToGG', 'VHToGG'],
    # 'GGJets': ['GGJets'], 
    # 'DDQCDGJets': ['DDQCDGJets'], 
    # 'TTGG': ['TTGG'], 
    # 'GluGluHToGG': ['GluGluHToGG'], 
    # 'VBFHToGG': ['VBFHToGG'], 
    # 'ttHToGG': ['ttHToGG'], 
    # 'bbHToGG': ['bbHToGG'], 
    # 'VHToGG': ['VHToGG']
}


def save_roc(fprs, roc_str: str, file_postfix: str=''):
    reftpr = lambda reffpr: BASE_TPR[(np.abs(fprs[-1] - reffpr)).argmin()]
    plt.figure(figsize=(10,8))
    for i, fpr in enumerate(fprs[:-1]):
        plt.plot(fpr, BASE_TPR, label=f"Fold {i} - AUROC = {trapezoid(BASE_TPR, fpr):.3f}", alpha=0.6, linestyle='--')
    plt.plot(fprs[-1], BASE_TPR, label=f"Merged - AUROC = {trapezoid(BASE_TPR, fprs[-1]):.3f}", alpha=0.75)
    vlines_x = [1e-2] if '_Boost_' in dfdataset_config else [1e-3]
    vlines_y = [reftpr(vline_x) for vline_x in vlines_x]
    plt.vlines(vlines_x, 1e-3, 1, linestyles='dotted', colors=['deeppink'], label=", ".join([r'$\epsilon_{sig}$ = '+f"{vlines_y[i]:.2e}"+r' @ $\epsilon_{bkg}$ = '+f"{vlines_x[i]:.2e}" for i in range(len(vlines_x))]))
    plt.title(roc_str.replace('  ', ' - '))
    plt.ylabel('Signal efficiency')
    plt.xlabel('Background efficiency')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e-6, 1))
    plt.ylim((1e-3, 1))
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{roc_str.replace(' ', '_')}_roc{file_postfix}.png"))
    plt.close()


def make_roc_curves(data: pd.DataFrame, fprs: dict, file_postfix: str=''):
    for signal, (signal_idx, signal_list) in signals.items():
        output_name = output_nodes[signal_idx].replace('AUX_', '')
        for bkg, bkg_list in bkgs.items():
            sub_data = data[data["AUX_sample_name"].isin(signal_list+bkg_list)]
            fpr, tpr, threshold = roc_curve(sub_data["AUX_label1D"].eq(signal_idx), sub_data[output_nodes[signal_idx]], sample_weight=sub_data["AUX_eventWeight"])
            fpr = np.interp(BASE_TPR, tpr, fpr)

            roc_str = f'{output_name}  {signal} vs {bkg}'
            if roc_str not in fprs.keys(): fprs[roc_str] = [fpr]
            else: fprs[roc_str].append(fpr)

fprs = {}
for fold_idx in range(dfdataset.n_folds):
    data = dfdataset.get_test(fold_idx, columns=columns_map)
    make_roc_curves(data, fprs)
data = dfdataset.get_all_test(columns=columns_map)
make_roc_curves(data, fprs)

for roc_str, fpr_list in fprs.items():
    save_roc(fpr_list, roc_str)