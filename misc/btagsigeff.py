import pyarrow.parquet as pq
import pandas as pd
import numpy as np


signal_files = {
    "2022preEE": "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/Run3_2022/sim/preEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
    "2022postEE": "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/Run3_2022/sim/postEE/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
    "2023preBPix": "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/Run3_2023/sim/preBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
    "2023postBPix": "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/Run3_2023/sim/postBPix/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
    "2024": "/eos/uscms/store/group/lpcdihiggsboost/tsievert/HiggsDNA_parquet/v4/Run3_2024/sim/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
}

for era, sigfile in signal_files.items():

    if "2024" in era: btagger, lead_btag_var, sublead_btag_var = "UParT", "lead_bjet_btagUParTAK4B", "sublead_bjet_btagUParTAK4B"
    else: btagger, lead_btag_var, sublead_btag_var = "PNet", "_lead_bjet_btagPNetB", "_sublead_bjet_btagPNetB"

    df = pq.read_table(sigfile).to_pandas()

    cuts_to_try = np.linspace(0.999, 0.9995, 1000)

    sig_effs = []
    for cut in cuts_to_try:
        sig_effs.append(
            np.sum(
                np.logical_or(df[f"nonRes{lead_btag_var}"] > cut, df[f"nonRes{sublead_btag_var}"] > cut)
            ) / np.sum(
                np.logical_or(df[f"nonRes{lead_btag_var}"] > 0., df[f"nonRes{sublead_btag_var}"] > 0.)
            )
        )

    sig_effs = np.array(sig_effs)

    best_idx = np.argmin(np.abs(sig_effs - 0.1))
    print(f"{era}: Closest eff = {sig_effs[best_idx]:.5f} with {btagger} cut at {cuts_to_try[best_idx]:.5f}")
