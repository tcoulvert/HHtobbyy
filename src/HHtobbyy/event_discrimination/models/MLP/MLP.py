# ML packages
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# # check
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# import torch

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models.MLP.MLPTorch import MLPTorch
from HHtobbyy.event_discrimination.models.MLP.MLPDataset import MLPDataset
from HHtobbyy.event_discrimination.models.MLP.MLPConfig import MLPConfig

################################


class MLP(Model):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        self.modeldataset = MLPDataset(self.dfdataset, config)
        self.modelconfig = MLPConfig(self.dfdataset, config)

    def load_model_and_trainer(self, ckpt_path: str='', eval: bool=False):
        # DNN model
        if ckpt_path != '':
            model = MLPTorch.load_from_checkpoint(
                ckpt_path, weights_only=False, 
                **{'input_size': len(self.dfdataset.model_vars), 'num_layers': self.modelconfig.num_layers, 'num_nodes': self.modelconfig.num_nodes, 'output_size': self.dfdataset.n_classes, 'dropout_prob': self.modelconfig.dropout_prob}
            )
        else:
            model = MLPTorch(
                len(self.dfdataset.model_vars), self.modelconfig.num_layers, self.modelconfig.num_nodes, self.dfdataset.n_classes, self.modelconfig.dropout_prob
            )

        # Callbacks
        callbacks = [EarlyStopping(monitor=self.modelconfig.monitor, min_delta=self.modelconfig.min_delta, patience=self.modelconfig.patience, verbose=False, mode=self.modelconfig.mode)]

        trainer = Trainer(
            callbacks=callbacks,
            default_root_dir=self.modelconfig.output_dirpath,
            max_epochs=self.modelconfig.max_epochs, 
            accelerator=self.modelconfig.accelerator,
            strategy=self.modelconfig.strategy,
            num_nodes=self.modelconfig.num_nodes,
            precision=self.modelconfig.precision, 
            gradient_clip_val=self.modelconfig.gradient_clip_val,
            logger=self.modelconfig.logger,
            devices=1 if eval else self.modelconfig.devices,
            enable_checkpointing=not eval
        )

        return model, trainer

    def train(self, fold: int, resume_from_ckpt: bool=False):
        # Data
        train_data = self.modeldataset.get_train(fold)
        val_data = self.modeldataset.get_val(fold)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer()

        # Train DNN
        trainer.fit(model, train_data, val_data, ckpt_path=self.modelconfig.get_ckpt_path(fold) if resume_from_ckpt else None)

    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train'):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer(ckpt_path=self.modelconfig.get_ckpt_path(fold), eval=True)

        # Test data predictions
        trainer.test(model, eval_data)
    
    def predict(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer(ckpt_path=self.modelconfig.get_ckpt_path(fold), eval=True)

        # Test data predictions
        predictions = trainer.predict(model, eval_data)
        
        return predictions