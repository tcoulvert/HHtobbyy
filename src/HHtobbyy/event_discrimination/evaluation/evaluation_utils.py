# Common Py packages
import numpy as np

# Workspace packages
from HHtobbyy.workspace_utils.retrieval_utils import format_class_names

################################


TRANSFORM_PREDS = [
    {
        'name': 'nD', 
        'output': lambda class_names: class_discriminator_columns(class_names), 
        'ROC_bkgeffs': lambda class_names: [1e-3 for _ in class_names],
        'func': lambda multibdt_output: multibdt_output,
        'cutdir': ['>', '<', '<', '<']
    },
    {
        'name': 'DttH-DQCD', 
        'output': lambda class_names: ['DttH', 'DQCD'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-3],
        'func': lambda multibdt_output: DttHDQCD(multibdt_output),
        'cutdir': ['>', '>']
    },
    {
        'name': '3D', 
        'output': lambda class_names: ['a', 'b', 'c'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-2, 1e-3],
        'func': lambda multibdt_output: abc(multibdt_output),
        'cutdir': ['<', '<', '<']
    },
]

################################


def discriminator_columns(columns: list[str]):
    return ['D'+column for column in columns]
def class_discriminator_columns(class_names: list[str]):
    return discriminator_columns(format_class_names(class_names))

def transform_preds_options():
    return [transformation['name'] for transformation in TRANSFORM_PREDS]

def transform_preds_bkgeffs(class_names: list, transform_name: str):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    ROC_bkgeffs = [transformation['ROC_bkgeffs'](class_names) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return ROC_bkgeffs

def transform_preds_func(class_names: list, transform_name: str):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    output, func, cutdir = [(transformation['output'](class_names), transformation['func'], transformation['cutdir']) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return output, func, cutdir

def transform_preds(class_names: list, transform_name: str, preds: np.ndarray):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    output, func, cutdir = [(transformation['output'](class_names), transformation['func'], transformation['cutdir']) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return output, func(preds), cutdir


def DttHDQCD(multibdt_output):
    DttH_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])
    DQCD_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])

    DttH_preds[np.isnan(DttH_preds)], DQCD_preds[np.isnan(DQCD_preds)] = 0, 0

    return np.column_stack([DttH_preds, DQCD_preds])

def abc(multibdt_output):
    a_preds = multibdt_output[:, 1]
    b_preds = multibdt_output[:, 2] / (1 - a_preds)
    c_preds = multibdt_output[:, 3] / ( (1 - a_preds) * (1-b_preds) )

    a_preds[np.isnan(a_preds)], b_preds[np.isnan(b_preds)], c_preds[np.isnan(c_preds)] = 0, 0, 0
    
    return np.column_stack([a_preds, b_preds, c_preds])

def b(multibdt_output):
    b_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])
    b_preds[np.isnan(b_preds)] = 0
    return b_preds

def DttH(multibdt_output):
    DttH_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])
    DttH_preds[np.isnan(DttH_preds)] = 0
    return DttH_preds
