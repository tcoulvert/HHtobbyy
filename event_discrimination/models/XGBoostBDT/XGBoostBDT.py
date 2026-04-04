# Stdlib packages
import datetime
import os

# ML packages
import xgboost as xgb

# HEP packages
import eos_utils as eos

# Workspace packages
from HHtobbyy.event_discrimination.dataset import DFDataset
from HHtobbyy.event_discrimination.models import Model
from HHtobbyy.event_discrimination.models.XGBoostBDT import XGBoostBDTDataset, XGBoostBDTConfig

################################


class XGBoostBDT(Model):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        self.modeldataset = XGBoostBDTDataset(self.dfdataset, config)
        self.modelconfig = XGBoostBDTConfig(self.dfdataset, config)

        # Current time of execution
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'

        # Output dirpath for model files
        self.output_dirpath = os.path.join(self.modelconfig.output_dirpath, self.dfdataset.dataset_tag, self.current_time)
        
        # General filenames
        self.model_filename = f'{self.current_time}_BDT_fold'
        self.eval_filename = f'{self.current_time}_eval_fold'

    def train(self, fold: int):
        # Save config
        self.modelconfig.save_config()
        
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

        booster.save_model(os.path.join(self.output_dirpath, f'{self.model_filename}{fold}.model'))
        eos.save_file_eos(eval_result, os.path.join(self.output_dirpath, f'{self.eval_filename}{fold}.json'))

    def evaluate(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        booster =  xgb.Booster(
            params=self.modelconfig.load_config(), 
            model_file=os.path.join(training_dirpath, f"{training_dirpath.split('/')[-2]}_BDT_fold{fold_idx}.model")
        )