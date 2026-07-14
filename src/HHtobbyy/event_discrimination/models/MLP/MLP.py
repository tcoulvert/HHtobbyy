# Stdlib packages
import os

# Common Py packages
import numpy as np

# ML packages
from lightning.pytorch.utilities.data import DataLoader
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import Model
from HHtobbyy.event_discrimination.models.MLP.MLPTorch import MLPTorch
from HHtobbyy.event_discrimination.models.MLP.MLPDataset import MLPDataset
from HHtobbyy.event_discrimination.models.MLP.MLPConfig import MLPConfig

################################


class MLP(Model):
    def __init__(self, dfdataset: str|DFDataset, config: str|dict):
        self.dfdataset = DFDataset(dfdataset) if type(dfdataset) is str else dfdataset
        self.modeldataset = MLPDataset(self.dfdataset, config)
        self.modelconfig = MLPConfig(self.dfdataset, config)

    def load_model_and_trainer(self, fold: int, n_batches: int, ckpt_path: str='', log_path: str='', eval: bool=False):
        # DNN model
        if ckpt_path != '': model = MLPTorch.load_from_checkpoint(ckpt_path, weights_only=False, **{**self.modelconfig.__dict__, **{'n_batches': n_batches}})
        else: model = MLPTorch(**{**self.modelconfig.__dict__, **{'n_batches': n_batches}})

        # Callbacks
        earlystopping_callback = EarlyStopping(
            monitor=self.modelconfig.monitor, min_delta=self.modelconfig.min_delta, 
            patience=self.modelconfig.patience, verbose=False, 
            mode=self.modelconfig.mode
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.modelconfig.output_dirpath, f"lightning_logs/version_{fold}/checkpoints"),
            monitor="val_loss", mode="min",
            filename="checkpoint-{epoch:02d}-{val_loss:.2f}", every_n_epochs=self.modelconfig.every_n_epochs, save_top_k=self.modelconfig.save_top_k
        )
        callbacks = [checkpoint_callback, earlystopping_callback]

        trainer = Trainer(
            callbacks=callbacks,
            default_root_dir=self.modelconfig.output_dirpath,
            accumulate_grad_batches=self.modelconfig.accumulate_grad_batches,
            max_epochs=self.modelconfig.max_epochs, 
            accelerator=self.modelconfig.accelerator,
            strategy=self.modelconfig.strategy,
            num_nodes=self.modelconfig.num_nodes,
            precision=self.modelconfig.precision, 
            gradient_clip_val=self.modelconfig.gradient_clip_val,
            logger=False if eval and log_path == '' else (log_path if log_path != '' else self.modelconfig.logger),
            devices=1 if eval else self.modelconfig.devices,
        )

        return model, trainer

    def train(self, fold: int, tune_lr: bool=False, resume_from_ckpt: bool=False, **kwargs):
        # Data
        train_data = self.modeldataset.get_train(fold)
        val_data = self.modeldataset.get_val(fold)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer(fold, n_batches=len(train_data))

        if tune_lr: tuner = Tuner(trainer); tuner.lr_find(model)

        # Train DNN
        trainer.fit(model, train_data, val_data, ckpt_path=self.modelconfig.get_ckpt_path(fold) if resume_from_ckpt else None)

    def test(self, fold: int, syst_name: str='nominal', regex: str|list[str]='test_of_train', **kwargs):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer(fold, n_batches=len(eval_data), ckpt_path=self.modelconfig.get_ckpt_path(fold), log_path=self.modelconfig.get_log_path(fold), eval=True)

        # Test data predictions
        trainer.test(model, eval_data)
        
    def predict_data(self, data: DataLoader, fold: int, ckpt_path: str='', **kwargs):
        # DNN model and trainer
        if ckpt_path == '': ckpt_path = self.modelconfig.get_ckpt_path(fold)
        model, trainer = self.load_model_and_trainer(fold, n_batches=len(data), ckpt_path=ckpt_path, eval=True)

        predictions = trainer.predict(model, data)
        predictions = np.concatenate([prediction.numpy(force=True) for prediction in predictions])

        return predictions
