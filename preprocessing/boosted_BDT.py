# Stdlib packages
import datetime

################################


CLASS_SAMPLE_MAP = {
    'ggF HH': ["GluGlu*HH*kl-1p00"],
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

    # lepton vars
    'lepton1_pt', 'lepton1_pfIsoId', 'lepton1_mvaID',

    # angular vars
    f'{jet_prefix}_CosThetaStar_CS', f'{jet_prefix}_CosThetaStar_gg',

    # fatjet vars
    
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
}
MHH_CORRELATED_VARIABLES = lambda jet_prefix: {
    # MHH
    # f'{jet_prefix}_HHbbggCandidate_mass',

    # MET variables
    'puppiMET_sumEt',  #eft

    # fatjet vars

    # diphoton vars
    'pt',  #eft
}
AUX_VARIABLES = lambda jet_prefix: {
    # identifiable event info
    'event', 'lumi', 'hash', 'sample_name', 

    # MC info
    'weight', 'eventWeight',

    # mass
    'mass', 
    f'{jet_prefix}_HHbbggCandidate_mass',

    # event masks
    f'{jet_prefix}_resolved_BDT_mask',
}

FILL_VALUE = -999
TRAIN_MOD = 5
JET_PREFIX = 'Res'

SEED = 21
BASE_FILEPATH = 'Run3_202'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DF_MASK = 'default'  # 'none'

END_FILEPATH = "preprocessed.parquet"

################################


BDT_VARIABLES = BASIC_VARIABLES(JET_PREFIX) | MHH_CORRELATED_VARIABLES(JET_PREFIX)
AUX_VARIABLES = AUX_VARIABLES(JET_PREFIX)