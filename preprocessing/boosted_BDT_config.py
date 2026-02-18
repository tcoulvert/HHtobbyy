# Stdlib packages
import datetime

################################


DATASET_TAG = "22to24_bTagWP"

CLASS_SAMPLE_MAP = {
    'ggF HH': ["GluGlu*HH*kl-1p00"],  # *!batch 
    **({'VBF HH': ["VBF*HH*C2V_1_"]} if 'VBFHH' in DATASET_TAG else {}),
    'ttH + bbH': ["ttH", "bbH"],
    'VH': ["VH", "ZH", "Wm*H", "Wp*H"],
    **(
        {'nonRes + ggFH + VBFH': ["!DDQCDGJet*GGJets", "!DDQCDGJet*GJet", "TTGG", "GluGluH*GG", "VBFH*GG"]} 
        if 'DDQCD' not in DATASET_TAG.upper() else {'nonRes + ggFH + VBFH': ["DDQCDGJets", "TTGG", "GluGluH*GG", "VBFH*GG"]}
    ),
    # 'nonRes + ggFH + VBFH': ["GJet", "TTGG", "GluGluH*GG", "VBFH*GG"],
}
TRAIN_ONLY_SAMPLES = {
    "Zto2Q", "Wto2Q", "batch[4-6]"
}
TEST_ONLY_SAMPLES = {
    "Data", "GluGlu*HH*kl-0p00", "GluGlu*HH*kl-2p45", "GluGlu*HH*kl-5p00", "SherpaNLO"
}

BASIC_VARIABLES = lambda jet_prefix: {
    # MET variables
    'puppiMET_pt',

    # lepton vars
    'lepton1_pt', 'lepton1_pfIsoId', 'lepton1_mvaID',

    # angular vars
    f'{jet_prefix}_CosThetaStar_CS', f'{jet_prefix}_CosThetaStar_gg',

    # fatjet vars
    'fatjet_selected_eta',  # eta
    'fatjet_selected_bbTagWP',  # bbTag
    'fatjet_selected_tau21', 'fatjet_selected_tau32',
    
    # diphoton vars
    'eta',
    # 'pt_Over_FatjetPt'

    # Photon vars
    'lead_mvaID', 'lead_sigmaE_over_E', 'deltaR_g1_fj',
    # --------
    'sublead_mvaID', 'sublead_sigmaE_over_E', 'deltaR_g2_fj',
    
    # HH vars
    f'{jet_prefix}_HHbbggCandidate_pt', 
    f'{jet_prefix}_HHbbggCandidate_eta', 
    f'{jet_prefix}_fatjet_pt_balance'
}
MHH_CORRELATED_VARIABLES = lambda jet_prefix: {
    # MHH
    f'{jet_prefix}_HHbbggCandidate_mass',

    # MET variables
    'puppiMET_sumEt',

    # fatjet vars
    'fatjet_selected_msoftdrop', 
    'fatjet_selected_pt',

    # diphoton vars
    'pt',
}
AUX_VARIABLES = lambda jet_prefix: {
    # identifiable event info
    'event', 'lumi', 'hash', 'sample_name', 'sample_era',

    # MC info
    'weight', 'eventWeight',

    # mass
    'mass',
    f'{jet_prefix}_HHbbggCandidate_mass',

    # sculpting study
    'max_nonselectedfatjet_bbtag',

    # event masks
    'boosted_BDT_mask',
}

FILL_VALUE = -999
TRAIN_MOD = 5
JET_PREFIX = 'Res'  # ["Res", "Res_DNNpair", "nonRes", "nonResReg", "nonResReg_DNNpair"]

SEED = 21
BASE_FILEPATH = 'Run3_20'
CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DF_MASK = 'default'  # 'none'

END_FILEPATH = "preprocessed.parquet"

################################


BDT_VARIABLES = BASIC_VARIABLES(JET_PREFIX) | MHH_CORRELATED_VARIABLES(JET_PREFIX)
AUX_VARIABLES = AUX_VARIABLES(JET_PREFIX)