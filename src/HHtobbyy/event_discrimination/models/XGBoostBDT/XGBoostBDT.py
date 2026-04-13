# Stdlib packages
import os

# ML packages
import xgboost as xgb

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models.XGBoostBDT.XGBoostBDTDataset import XGBoostBDTDataset
from HHtobbyy.event_discrimination.models.XGBoostBDT.XGBoostBDTConfig import XGBoostBDTConfig

################################


# https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
#  -> For tne next iteration of BDT?
class XGBoostBDT(Model):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        self.modeldataset = XGBoostBDTDataset(self.dfdataset, config)
        self.modelconfig = XGBoostBDTConfig(self.dfdataset, config)
        
        # General filenames
        self.model_filename = f'{self.modelconfig.model_time}_BDT_fold'
        self.eval_filename = f'{self.modelconfig.model_time}_eval_fold'

        # Save config
        self.modelconfig.save_config()

    def train(self, fold: int):
        # Data
        train_data = self.modeldataset.get_train(fold)
        val_data = self.modeldataset.get_val(fold)

        # Train BDT
        eval_result = {}
        evallist = [(train_data, 'train'), (val_data, 'val')]
        booster = xgb.train(
            self.modelconfig.__dict__, train_data, num_boost_round=self.modelconfig.num_trees, 
            evals=evallist, early_stopping_rounds=10, 
            verbose_eval=self.modelconfig.verbose_eval, evals_result=eval_result,
        )

        booster.save_model(os.path.join(self.modelconfig.output_dirpath, f'{self.model_filename}{fold}.model'))
        eos.save_file_eos(eval_result, os.path.join(self.modelconfig.output_dirpath, f'{self.eval_filename}{fold}.json'))

    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train'):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)

        # Initialize trained BDT model
        booster = xgb.Booster(
            params=self.modelconfig.load_config(), 
            model_file=os.path.join(self.modelconfig.output_dirpath, f"{self.model_filename}{fold}.model")
        )

        # Test data predictions
        predictions = booster.predict(eval_data, iteration_range=(0, booster.best_iteration))
        # loss = predictions - eval_data[label]

    def predict_data(self, data: xgb.DMatrix, fold: int):
        # Initialize trained BDT model
        booster = xgb.Booster(
            params=self.modelconfig.load_config(), 
            model_file=os.path.join(self.modelconfig.output_dirpath, f"{self.model_filename}{fold}.model")
        )

        # Test data predictions
        predictions = booster.predict(data, iteration_range=(0, booster.best_iteration))

        return predictions
    
    def predict(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)
        return self.predict_data(eval_data, fold)

        