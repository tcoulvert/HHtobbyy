from HHtobbyy.event_discrimination.models.Model import Model
from HHtobbyy.event_discrimination.models.ModelConfig import ModelConfig

from HHtobbyy.event_discrimination.models.MLP.MLP import MLP
from HHtobbyy.event_discrimination.models.XGBoostBDT.XGBoostBDT import XGBoostBDT


def map_model_to_Model(model_name: str):
    if model_name.lower() == "XGBoostBDT".lower(): return XGBoostBDT
    elif model_name.lower() == "MLP".lower(): return MLP
    else: raise NotImplementedError(f"Requested model ({model_name}) not implemented yet, or if it is implemented map function hasn't been updated.")