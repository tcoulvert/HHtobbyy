# Stdlib packages
from abc import ABC, abstractmethod
import gc
import time
import tracemalloc

# Common Py packages
import numpy as np
import pandas as pd

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelConfig, ModelDataset
from HHtobbyy.event_discrimination.evaluation.evaluation_utils import class_discriminator_columns
from HHtobbyy.workspace_utils import multifold

################################


class Model(ABC):
    dfdataset: DFDataset
    modeldataset: ModelDataset
    modelconfig: ModelConfig

    def train_all_folds(self, **kwargs) -> None:
        multifold(self.train, (), self.dfdataset.n_folds, **kwargs)

    def test_all_folds(self, syst_name: str='nominal', regex: str|list[str]='', **kwargs) -> None:
        multifold(self.test, (syst_name, regex), self.dfdataset.n_folds, **kwargs)

    def predict_all_folds(self, syst_name: str='nominal', regex: str|list[str]='', ckpt_path: str='', **kwargs) -> None:
        multifold(self.predict, (syst_name, regex, ckpt_path), self.dfdataset.n_folds, **kwargs)

    @abstractmethod
    def train(self, fold: int) -> None:
        pass

    @abstractmethod
    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train') -> None:
        pass

    @abstractmethod
    def predict_data(self, data: object, fold: int, ckpt_path: str='') -> np.ndarray:
        pass

    def predict(self, fold: int, syst_name: str='nominal', regex: str|list[str]='', ckpt_path: str=''):
        tracemalloc.start()

        current, peak = tracemalloc.get_traced_memory()
        print(f"Pre-getting filelist memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)
        test_filepaths = self.dfdataset.get_test_filepaths(fold, syst_name=syst_name, regex=regex)['test']
        for filepath in test_filepaths:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Pre-loading file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            df = self.dfdataset.get_df(filepath)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Post-loading DF file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            # data = self.modeldataset.get_data(df, self.dfdataset.event_weight_var)
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Post-loading data memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            # predictions = self.predict_data(data, fold, ckpt_path=ckpt_path)
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Post-prediction file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)
            # predictions = pd.DataFrame(predictions, columns=[self.dfdataset.aux_var_prefix+col for col in class_discriminator_columns(self.dfdataset.class_sample_map.keys())])

            # # predictions = pd.DataFrame(
            # #     self.predict_data(
            # #         self.modeldataset.get_data(
            # #             self.dfdataset.get_df(filepath), 
            # #             self.dfdataset.event_weight_var
            # #         ), 
            # #         fold, ckpt_path=ckpt_path
            # #     ),
            # #     columns=[self.dfdataset.aux_var_prefix+col for col in class_discriminator_columns(self.dfdataset.class_sample_map.keys())]
            # # )

            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Post-building prediction DF file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            # try: self.dfdataset.save_df(filepath, predictions)
            # except: eos.save_file_eos(predictions, filepath.replace('.parquet', '_eval.parquet'), force=True)
            # del predictions

            current, peak = tracemalloc.get_traced_memory()
            print(f"Post-forloop file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Post-gc file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)

            # time.sleep(5)
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Post-sleep(5) file memory usage, peak memory usage: {current / 10**6:.2f}, {peak / 10**6:.2f} MB", '\n', '-'*60)
