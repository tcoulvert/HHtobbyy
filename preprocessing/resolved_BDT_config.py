# Stdlib packages
import datetime

################################


DATASET_TAG = "24_bTagWPbatch"

CLASS_SAMPLE_MAP = {
    'ggF HH': ["GluGlu*HH*kl-1p00"],  # !batch
    'ttH + bbH': ["ttH", "bbH"],
    'VH': ["VH", "ZH", "Wm*H", "Wp*H"],
    'nonRes + ggFH + VBFH': ["GGJets", "GJet", "TTGG", "GluGluH*GG", "VBFH*GG"]
}
TRAIN_ONLY_SAMPLES = {
    "Zto2Q", "Wto2Q", "batch[4-6]"
}
TEST_ONLY_SAMPLES = {
    "Data", "GluGlu*HH*kl-0p00", "GluGlu*HH*kl-2p45", "GluGlu*HH*kl-5p00"
}

BASIC_VARIABLES = lambda jet_prefix: {
    # MET variables
    'puppiMET_pt',

    # jet vars
    f'{jet_prefix}_DeltaPhi_j1MET', f'{jet_prefix}_DeltaPhi_j2MET', 
    'n_jets', f'{jet_prefix}_chi_t0', f'{jet_prefix}_chi_t1',

    # lepton vars
    'lepton1_pt', 'lepton1_pfIsoId', 'lepton1_mvaID',
    'DeltaR_b1l1',

    # angular vars
    f'{jet_prefix}_CosThetaStar_CS', f'{jet_prefix}_CosThetaStar_jj', f'{jet_prefix}_CosThetaStar_gg',

    # bjet vars
    f'{jet_prefix}_lead_bjet_eta', # eta
    # f"{jet_prefix}_lead_bjet_btagPNetB",
    # f"{jet_prefix}lead_bjet_btagUParTAK4B",
    f"{jet_prefix}_lead_bjet_bTagWPL", f"{jet_prefix}_lead_bjet_bTagWPM", f"{jet_prefix}_lead_bjet_bTagWPT",
    f"{jet_prefix}_lead_bjet_bTagWPXT", f"{jet_prefix}_lead_bjet_bTagWPXXT",
    # f"{jet_prefix}_lead_bjet_bTagWP3XT", f"{jet_prefix}_lead_bjet_bTagWP4XT",
    # --------
    f'{jet_prefix}_sublead_bjet_eta', 
    # f"{jet_prefix}_sublead_bjet_btagPNetB",
    # f"{jet_prefix}sublead_bjet_btagUParTAK4B",
    f"{jet_prefix}_sublead_bjet_bTagWPL", f"{jet_prefix}_sublead_bjet_bTagWPM", f"{jet_prefix}_sublead_bjet_bTagWPT",
    f"{jet_prefix}_sublead_bjet_bTagWPXT", f"{jet_prefix}_sublead_bjet_bTagWPXXT",
    # f"{jet_prefix}_sublead_bjet_bTagWP3XT", f"{jet_prefix}_sublead_bjet_bTagWP4XT",
    
    # diphoton vars
    'eta',

    # Photon vars
    'lead_mvaID', 'lead_sigmaE_over_E',
    # --------
    'sublead_mvaID', 'sublead_sigmaE_over_E',
    
    # HH vars
    f'{jet_prefix}_HHbbggCandidate_pt', 
    f'{jet_prefix}_HHbbggCandidate_eta', 
    f'{jet_prefix}_pt_balance',

    # ZH vars
    f'{jet_prefix}_DeltaPhi_jj',
    f'{jet_prefix}_DeltaPhi_isr_jet_z',
}
MHH_CORRELATED_VARIABLES = lambda jet_prefix: {
    # MHH
    # f'{jet_prefix}_HHbbggCandidate_mass',

    # MET variables
    'puppiMET_sumEt',  #eft

    # jet vars
    f'{jet_prefix}_DeltaR_jg_min',  #eft

    # dijet vars
    f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),  #eft
    f'{jet_prefix}_dijet_pt',  #eft

    # bjet vars
    f'{jet_prefix}_lead_bjet_pt', #eft
    f'{jet_prefix}_lead_bjet_sigmapT_over_pT', #eft
    f'{jet_prefix}_lead_bjet_pt_over_Mjj', #eft
    # --------
    f'{jet_prefix}_sublead_bjet_pt', #eft
    f'{jet_prefix}_sublead_bjet_sigmapT_over_pT', #eft
    f'{jet_prefix}_sublead_bjet_pt_over_Mjj', #eft

    # diphoton vars
    'pt',  #eft

    # ZH vars
    f'{jet_prefix}_DeltaEta_jj', #eft
    f'{jet_prefix}_isr_jet_pt',  #eft
}
AUX_VARIABLES = lambda jet_prefix: {
    # identifiable event info
    'event', 'lumi', 'hash', 'sample_name', 

    # MC info
    'weight', 'eventWeight',

    # mass
    'mass', 
    f'{jet_prefix}_dijet_mass' + ('' if jet_prefix == 'nonRes' else '_DNNreg'),
    f'{jet_prefix}_HHbbggCandidate_mass',

    # sculpting study
    f'{jet_prefix}_max_nonbjet_btag',

    # event masks
    f'{jet_prefix}_resolved_BDT_mask',
}

FILL_VALUE = -999
TRAIN_MOD = 5
JET_PREFIX = 'nonRes'

SEED = 21
BASE_FILEPATH = 'Run3_202'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DF_MASK = 'default'  # 'none'

END_FILEPATH = "preprocessed.parquet"

################################


BDT_VARIABLES = BASIC_VARIABLES(JET_PREFIX) | MHH_CORRELATED_VARIABLES(JET_PREFIX)
AUX_VARIABLES = AUX_VARIABLES(JET_PREFIX)