# Stdlib packages
import datetime
import os

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

        # Current time of execution
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 'YYYY-MM-DD_HH-MM-SS'

        # Output dirpath for model files
        self.output_dirpath = os.path.join(self.modelconfig.output_dirpath, self.dfdataset.dataset_tag, self.current_time)

    def train(self, fold: int):
        # Save config
        self.modelconfig.save_config()
        
        # Data
        train_data = self.modeldataset.get_train(fold)
        val_data = self.modeldataset.get_val(fold)

        # DNN model
        model = MLPTorch(train_data.shape[1], self.modelconfig.num_layers, self.modelconfig.num_nodes, self.dfdataset.n_classes, self.modelconfig.dropout_prob)

        # Callbacks
        callbacks = [EarlyStopping(monitor=self.modelconfig.monitor, min_delta=self.modelconfig.min_delta, patience=self.modelconfig.patience, verbose=False, mode=self.modelconfig.mode)]

        # Build trainer
        trainer = Trainer(
            callbacks=callbacks,
            default_root_dir=self.output_dirpath,
            max_epochs=self.modelconfig.max_epochs, 
            accelerator=self.modelconfig.accelerator,
            strategy=self.modelconfig.strategy,
            num_nodes=self.modelconfig.num_nodes,
            precision=self.modelconfig.precision, 
            gradient_clip_val=self.modelconfig.gradient_clip_val,
            logger=self.modelconfig.logger
        )

        # Train DNN
        trainer.fit(model, train_data, val_data)

    def evaluate(self, fold: int, syst_name: str='nominal', regex: str|list[str]=''):
        pass