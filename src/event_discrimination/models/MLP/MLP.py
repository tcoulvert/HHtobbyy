# ML packages
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Workspace packages
from HHtobbyy.event_discrimination.dataset import DFDataset
from HHtobbyy.event_discrimination.models import Model
from HHtobbyy.event_discrimination.models.MLP import MLPTorch, MLPDataset, MLPConfig

################################


class MLP(Model):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset
        self.modeldataset = MLPDataset(self.dfdataset, config)
        self.modelconfig = MLPConfig(self.dfdataset, config)
        
        # Save config
        self.modelconfig.save_config()

    def load_model_and_trainer(self, eval: bool=False):
        # DNN model
        model = MLPTorch(len(self.dfdataset.model_vars), self.modelconfig.num_layers, self.modelconfig.num_nodes, self.dfdataset.n_classes, self.modelconfig.dropout_prob)

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
            devices=1 if eval else 'auto',
        )

        return model, trainer

    def train(self, fold: int):
        # Data
        train_data = self.modeldataset.get_train(fold)
        val_data = self.modeldataset.get_val(fold)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer()

        # Train DNN
        trainer.fit(model, train_data, val_data)

    def evaluate(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        eval_data = self.modeldataset.get_test(fold, syst_name=syst_name, regex=regex)

        # DNN model and trainer
        model, trainer = self.load_model_and_trainer(eval=True)

        # Test data predictions
        predictions = trainer.predict(model, eval_data, ckpt_path="best")
        print(predictions)
        
        return predictions