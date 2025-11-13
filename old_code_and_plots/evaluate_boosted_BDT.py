import os
import re
import joblib

import pandas as pd
import numpy as np
import awkward as ak

import xgboost as xgb
import vector


# ----------------------
# Main part of the file, single parquet having proc 
# ----------------------

def evaluate_boosted(
    sample, 
    boosted_model_filepaths: dict = {
        'all_plus_vh': 'clf_all_plus_vh_separateParquet.pkl',
        'vh_model': 'clf_vh_model_multiclassSeparateParquets.pkl'
    }
):
    all_columns = sample.fields

    # Filter for columns related to fatjets.
    fatjet_columns = [col for col in all_columns if 'fatjet' in col]
    nonResReg_columns = [col for col in all_columns if 'nonResReg' in col]
    weight_columns = [col for col in all_columns if 'weight' in col]
    sample_columns = [col for col in all_columns if 'sample' in col]
    # Essential columns needed for analysis and plotting.
    essential_columns = [
        'mass',
        'pt',
        'eta',
        'phi',
        'lead_eta',
        'lead_phi',
        'sublead_eta',
        'sublead_phi',
        'n_fatjets',
        'n_leptons',
        'n_jets',
        'lead_mvaID',
        'sublead_mvaID',
    ]

    bad_columns = [col for col in all_columns if len(np.shape(sample[col])) > 1]

    # Combine fatjet columns with essential columns.
    columns_to_load = list(
        set(fatjet_columns + essential_columns + nonResReg_columns + weight_columns + sample_columns)
        - set(bad_columns)
    )
    
    sample_dict = {}
    for field in columns_to_load:
        sample_dict[field] = ak.to_numpy(sample[field], allow_missing=False)
        if sample_dict[field].dtype == np.float64:
            sample_dict[field] = np.array(sample_dict[field], dtype=np.float32)
        if len(sample_dict[field].shape) > 1:
            print(field)
            print(sample_dict[field].shape)
    df = pd.DataFrame(sample_dict)
    print(f"Successfully loaded filtered data with {len(df)} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # ---------------------------------
    # Vectorized fatjet pre-selections
    # ---------------------------------

    # Identify and sort all msoftdrop columns by fatjet index.
    mass_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_msoftdrop', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_msoftdrop', x).group(1))
    )
    print("Finished softdrop mass sorting")
    if len(mass_cols) == 0:
        raise ValueError("No fatjet mass columns found.")

    mass_arr = df[mass_cols].to_numpy()
    eligible = (mass_arr > 30) & (mass_arr < 210)

    subjet1_eta_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_subjet1_eta', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_subjet1_eta', x).group(1))
    )
    subjet2_eta_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_subjet2_eta', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_subjet2_eta', x).group(1))
    )
    subjet1_eta_arr = df[subjet1_eta_cols].to_numpy()
    subjet2_eta_arr = df[subjet2_eta_cols].to_numpy()
    eligible = eligible & (subjet1_eta_arr != -999) & (subjet2_eta_arr != -999)

    eta_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_eta', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_eta', x).group(1))
    )
    eta_arr = df[eta_cols].to_numpy()
    eligible = eligible & (np.abs(eta_arr) <= 2.4)

    # Apply tau ratio cut: require fatjet tau2/tau1 < 0.75
    tau1_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_tau1', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_tau1', x).group(1))
    )
    tau2_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_tau2', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_tau2', x).group(1))
    )
    tau1_arr = df[tau1_cols].to_numpy()  # shape: (num_events, num_fatjets)
    tau2_arr = df[tau2_cols].to_numpy()  # shape: (num_events, num_fatjets)
    # Avoid division by zero: set ratio to infinity where tau1==0.
    FALLBACK_VALUE = 999.  # or any large value that definitely won't pass the cut
    tau_ratio_arr = np.full_like(tau1_arr, FALLBACK_VALUE, dtype=np.float32)
    nonzero_mask = tau1_arr != 0
    tau_ratio_arr[nonzero_mask] = tau2_arr[nonzero_mask] / tau1_arr[nonzero_mask]
    eligible = eligible & (tau_ratio_arr < 0.75)

    # Identify and sort all particleNet_XbbVsQCD columns by fatjet index.
    particleNet_cols = sorted(
        [col for col in df.columns if re.match(r'fatjet\d+_particleNet_XbbVsQCD', col)],
        key=lambda x: int(re.search(r'fatjet(\d+)_particleNet_XbbVsQCD', x).group(1))
    )
    if len(particleNet_cols) == 0:
        raise ValueError("No fatjet particleNet_XbbVsQCD columns found.")

    # Create a NumPy array for the particleNet scores.
    particleNet_arr = df[particleNet_cols].to_numpy()  # shape: (num_events, num_fatjets)
    eligible = eligible & (particleNet_arr > 0.4)

    # Mask non-eligible fatjets by replacing their particleNet scores with -infinity.
    masked_scores = np.where(eligible, particleNet_arr, -np.inf)

    # For each event, select the fatjet index with the highest particleNet score among eligible jets.
    best_idx = np.argmax(masked_scores, axis=1)

    n_fatjets_final = eligible.sum(axis=1)
    df['n_fatjets_final'] = n_fatjets_final

    # Build a dictionary mapping each fatjet property to its corresponding columns.
    prop_dict = {}
    pattern = re.compile(r'fatjet(\d+)_(.+)')
    for col in df.columns:
        m = pattern.match(col)
        if m:
            jet_index = int(m.group(1))
            prop = m.group(2)
            prop_dict.setdefault(prop, []).append((jet_index, col))

    # For each property, create a new 'fatjet_selected_{prop}' column by taking the value
    # from the column corresponding to the best (highest particleNet score) eligible fatjet.
    for prop, jets in prop_dict.items():
        jets_sorted = sorted(jets, key=lambda x: x[0])
        col_names = [col for (_, col) in jets_sorted]
        values = df[col_names].to_numpy()
        selected_vals = np.take_along_axis(values, best_idx[:, np.newaxis], axis=1).flatten()
        df[f'fatjet_selected_{prop}'] = selected_vals

    print(f"After msoftdrop cut and best particleNet selection, data has {len(df)} rows")
    df['fatjet_selected_tau2tau1_ratio'] = df['fatjet_selected_tau2'] / df['fatjet_selected_tau1']

    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    print("Define shorthand references")
    g1_eta  = df['lead_eta']
    g1_phi  = df['lead_phi']
    g2_eta  = df['sublead_eta']
    g2_phi  = df['sublead_phi']
    gg_eta  = df['eta']  # entire diphoton system
    gg_phi  = df['phi']  # entire diphoton system
    fj_eta  = df['fatjet_selected_eta']
    fj_phi  = df['fatjet_selected_phi']
    subj1_eta = df['fatjet_selected_subjet1_eta']
    subj1_phi = df['fatjet_selected_subjet1_phi']
    subj2_eta = df['fatjet_selected_subjet2_eta']
    subj2_phi = df['fatjet_selected_subjet2_phi']

    df['deltaEta_g1_g2'] = g1_eta - g2_eta
    df['deltaPhi_g1_g2'] = wrap_angle(g1_phi - g2_phi)

    df['deltaEta_gg_fj'] = gg_eta - fj_eta
    df['deltaPhi_gg_fj'] = wrap_angle(gg_phi - fj_phi)
    df['deltaR_gg_fj'] = np.sqrt(df['deltaEta_gg_fj']**2 + df['deltaPhi_gg_fj']**2)

    df['deltaEta_g1_fj'] = g1_eta - fj_eta
    df['deltaPhi_g1_fj'] = wrap_angle(g1_phi - fj_phi)
    df['deltaR_g1_fj'] = np.sqrt(df['deltaEta_g1_fj']**2 + df['deltaPhi_g1_fj']**2)

    df['deltaEta_g2_fj'] = g2_eta - fj_eta
    df['deltaPhi_g2_fj'] = wrap_angle(g2_phi - fj_phi)
    df['deltaR_g2_fj'] = np.sqrt(df['deltaEta_g2_fj']**2 + df['deltaPhi_g2_fj']**2)

    df['deltaEta_subj1_gg'] = subj1_eta - gg_eta
    df['deltaPhi_subj1_gg'] = wrap_angle(subj1_phi - gg_phi)
    df['deltaR_subj1_gg'] = np.sqrt(df['deltaEta_subj1_gg']**2 + df['deltaPhi_subj1_gg']**2)

    df['deltaEta_subj2_gg'] = subj2_eta - gg_eta
    df['deltaPhi_subj2_gg'] = wrap_angle(subj2_phi - gg_phi)
    df['deltaR_subj2_gg'] = np.sqrt(df['deltaEta_subj2_gg']**2 + df['deltaPhi_subj2_gg']**2)

    df['deltaEta_subj1_subj2'] = subj1_eta - subj2_eta
    df['deltaPhi_subj1_subj2'] = wrap_angle(subj1_phi - subj2_phi)
    df['deltaR_subj1_subj2'] = np.sqrt(df['deltaEta_subj1_subj2']**2 + df['deltaPhi_subj1_subj2']**2)

    df['deltaEta_g1_subj1'] = g1_eta - subj1_eta
    df['deltaPhi_g1_subj1'] = wrap_angle(g1_phi - subj1_phi)
    df['deltaR_g1_subj1'] = np.sqrt(df['deltaEta_g1_subj1']**2 + df['deltaPhi_g1_subj1']**2)

    df['deltaEta_g1_subj2'] = g1_eta - subj2_eta
    df['deltaPhi_g1_subj2'] = wrap_angle(g1_phi - subj2_phi)
    df['deltaR_g1_subj2'] = np.sqrt(df['deltaEta_g1_subj2']**2 + df['deltaPhi_g1_subj2']**2)

    df['deltaEta_g2_subj1'] = g2_eta - subj1_eta
    df['deltaPhi_g2_subj1'] = wrap_angle(g2_phi - subj1_phi)
    df['deltaR_g2_subj1'] = np.sqrt(df['deltaEta_g2_subj1']**2 + df['deltaPhi_g2_subj1']**2)

    df['deltaEta_g2_subj2'] = g2_eta - subj2_eta
    df['deltaPhi_g2_subj2'] = wrap_angle(g2_phi - subj2_phi)
    df['deltaR_g2_subj2'] = np.sqrt(df['deltaEta_g2_subj2']**2 + df['deltaPhi_g2_subj2']**2)

    df['DeltaR_jg_min'] = df[["deltaR_g1_subj1", "deltaR_g1_subj2", "deltaR_g2_subj1", "deltaR_g2_subj2"]].min(axis=1)

    xbb = df['fatjet_selected_particleNet_XbbVsQCD']
    df['fatjet_selected_Xbb_wp2'] = ((xbb >= 0.95) & (xbb < 0.975)).astype(int)
    df['fatjet_selected_Xbb_wp3'] = ((xbb >= 0.975) & (xbb < 0.99)).astype(int)
    df['fatjet_selected_Xbb_wp4'] = (xbb >= 0.99).astype(int)

    features = [
        'fatjet_selected_msoftdrop',
        'fatjet_selected_tau2tau1_ratio',
        'fatjet_selected_Xbb_wp2',
        'fatjet_selected_Xbb_wp3',
        'fatjet_selected_Xbb_wp4',
        'fatjet_selected_pt',
        'lead_eta',
        'sublead_eta',
        'eta',
        'n_leptons',
        'sublead_mvaID',
        'lead_mvaID',
        'nonResReg_CosThetaStar_gg',
        'nonResReg_pholead_PtOverM',
        'nonResReg_phosublead_PtOverM',
        'DeltaR_jg_min',
        'deltaEta_g1_g2',
        'deltaPhi_g1_g2',
        'deltaEta_gg_fj',
        'deltaPhi_gg_fj',
        'deltaR_gg_fj',
        'deltaEta_g1_fj',
        'deltaPhi_g1_fj',
        'deltaR_g1_fj',
        'deltaEta_g2_fj',
        'deltaPhi_g2_fj',
        'deltaR_g2_fj',
        'deltaEta_subj1_gg',
        'deltaPhi_subj1_gg',
        'deltaR_subj1_gg',
        'deltaEta_subj2_gg',
        'deltaPhi_subj2_gg',
        'deltaR_subj2_gg',
        'deltaEta_subj1_subj2',
        'deltaPhi_subj1_subj2',
        'deltaR_subj1_subj2',
        'deltaEta_g1_subj1',
        'deltaPhi_g1_subj1',
        'deltaR_g1_subj1',
        'deltaEta_g1_subj2',
        'deltaPhi_g1_subj2',
        'deltaR_g1_subj2',
        'deltaEta_g2_subj1',
        'deltaPhi_g2_subj1',
        'deltaR_g2_subj1',
        'deltaEta_g2_subj2',
        'deltaPhi_g2_subj2',
        'deltaR_g2_subj2',
        'n_jets',
        'n_fatjets',
        'n_fatjets_final'
    ]

    X = df[features]

    clf_vh_loaded = joblib.load(boosted_model_filepaths['vh_model'])
    try:
        model = joblib.load(boosted_model_filepaths['vh_model'])
        print("Model loading done. Type:", type(model))
    except Exception as e:
        print("Error in loading the model:", e)

    vh_score = clf_vh_loaded.predict_proba(X)
    vh_score = clf_vh_loaded.predict_proba(X)[:, 1]
    X = X.copy()
    X['vh_score'] = vh_score

    clf_all_loaded = joblib.load(boosted_model_filepaths['all_plus_vh'])
    y_proba = clf_all_loaded.predict_proba(X)[:, 1]

    df.loc[:, 'vh_score'] = vh_score
    df.loc[:, 'y_proba'] = y_proba

    vec1 = vector.arr({
        "pt": df["fatjet_selected_pt"],
        "eta": df["fatjet_selected_eta"],
        "phi": df["fatjet_selected_phi"],
        "mass": df["fatjet_selected_msoftdrop"]
    })
    vec2 = vector.arr({
        "pt": df["pt"],
        "eta": df["eta"],
        "phi": df["phi"],
        "mass": df["mass"]
    })
    df.loc[:, "mass_HH"] = (vec1 + vec2).mass

    df.loc[df["n_fatjets_final"] <= 0, "y_proba"] = -99

    BDT_boosted_cut = 0.856
    df['is_boosted'] = df['y_proba'] >= BDT_boosted_cut

    sample['is_boosted'] = df['is_boosted']
    sample['y_proba'] = df['y_proba']
    sample['vh_score'] = df['vh_score']
