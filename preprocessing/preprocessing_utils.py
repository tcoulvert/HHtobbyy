# Stdlib packages
import math

# HEP packages
import awkward as ak
import vector as vec

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import FILL_VALUE

################################


def ak_sign(ak_array, inverse=False):
    if not inverse:
        return ak.where(ak_array < 0, -1, 1)
    else:
        return ak.where(ak_array < 0, 1, -1)
        
def ak_abs(ak_array):
    valid_entry_mask = ak.where(ak_array != FILL_VALUE, True, False)
    abs_ak_array = ak.where(ak_array > 0, ak_array, -ak_array)
    return ak.where(valid_entry_mask, abs_ak_array, FILL_VALUE)
    
def deltaPhi(phi1, phi2):
    # angle1 and angle2 are (-pi, pi]
    # Convention: clockwise is (+), anti-clockwise is (-)
    subtract_angles = phi1 - phi2
    return ak.where(ak_abs(subtract_angles) <= math.pi, subtract_angles, subtract_angles + 2*math.pi*ak_sign(subtract_angles, inverse=True))

def deltaEta(eta1, eta2):
    return ak_abs(eta1 - eta2)
     