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
        'cutdir': lambda class_names: ['>']+['<' for _ in range(1, len(class_names))]
    },
    {
        'name': 'DttH-DQCD', 
        'output': lambda class_names: ['DttH', 'DQCD'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-3],
        'func': lambda multibdt_output: DttHDQCD(multibdt_output),
        'cutdir': lambda class_names: ['>', '>']
    },
    {
        'name': '3D', 
        'output': lambda class_names: ['a', 'b', 'c'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-2, 1e-3],
        'func': lambda multibdt_output: abc(multibdt_output),
        'cutdir': lambda class_names: ['<', '<', '<']
    },
    {
        'name': '4D', 
        'output': lambda class_names: ['A', 'B', 'C', 'D'], 
        'ROC_bkgeffs': lambda class_names: [1e-2, 1e-2, 1e-2, 1e-3],
        'func': lambda multibdt_output: ABCD(multibdt_output),
        'cutdir': lambda class_names: ['<', '<', '<', '<']
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
    
    output, func, cutdir = [(transformation['output'](class_names), transformation['func'], transformation['cutdir'](class_names)) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return output, func, cutdir

def transform_preds(class_names: list, transform_name: str, preds: np.ndarray):
    if transform_name not in transform_preds_options():
        raise KeyError(f"Output transformation {transform_name} not implemented, try one of {transform_preds_options()}")
    
    output, func, cutdir = [(transformation['output'](class_names), transformation['func'], transformation['cutdir'](class_names)) for transformation in TRANSFORM_PREDS if transform_name == transformation['name']][0]
    return output, func(preds), cutdir


def DttHDQCD(multibdt_output):
    DttH_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 1])
    DQCD_preds = multibdt_output[:, 0] / (multibdt_output[:, 0] + multibdt_output[:, 2] + multibdt_output[:, 3])

    DttH_preds[np.isnan(DttH_preds)], DQCD_preds[np.isnan(DQCD_preds)] = 0, 0

    return np.column_stack([DttH_preds, DQCD_preds])

def abc(multibdt_output):
    a_preds = multibdt_output[:, 1]
    b_preds = multibdt_output[:, 2] / (1 - a_preds)
    c_preds = multibdt_output[:, 3] / ( (1 - a_preds) * (1 - b_preds) )

    a_preds = np.nan_to_num(a_preds, copy=False)
    b_preds = np.nan_to_num(b_preds, copy=False)
    c_preds = np.nan_to_num(c_preds, copy=False)
    
    return np.column_stack([a_preds, b_preds, c_preds])

def ABCD(multibdt_output):
    A_preds = multibdt_output[:, 1]
    B_preds = multibdt_output[:, 2] / (1 - A_preds)
    C_preds = multibdt_output[:, 3] / ( (1 - A_preds) * (1 - B_preds) )
    D_preds = multibdt_output[:, 4] / ( (1 - A_preds) * (1 - B_preds) * (1 - C_preds))

    A_preds = np.nan_to_num(A_preds, copy=False)
    B_preds = np.nan_to_num(B_preds, copy=False)
    C_preds = np.nan_to_num(C_preds, copy=False)
    D_preds = np.nan_to_num(D_preds, copy=False)

    return np.column_stack([A_preds, B_preds, C_preds, D_preds])
