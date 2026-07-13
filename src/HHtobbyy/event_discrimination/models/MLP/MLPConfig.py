# Stdlib packages
import glob
import os

# ML packages
import gpustat

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.Model import ModelConfig

################################


class MLPConfig(ModelConfig):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset

        # Cross-saved parameters
        self.input_size        = len(dfdataset.model_vars)
        self.output_size       = dfdataset.n_classes

        # Required parameters
        self.output_dirpath    = ''

        # Architecture parameters
        self.num_layers        = 5           # number of hidden layers
        self.hidden_dim        = 1024        # dimensionality of hidden layers
        self.dropout_prob      = 0.25        # probability of dropping connections
        self.activation_func   = 'GELU'      # activation function for nonlinearity
        self.accumulate_grad_batches = 1     # number of batches to wait before stepping loss function
        self.learning_rate     = 1e-5        # learning rate for optimization

        # Dataset parameters
        self.class_weights     = None        # Extra class-weighting for training (can be done via DFDataset, which is preferred)

        # Eary stopping parameters
        self.min_delta         = 0.          # smallest val_loss difference
        self.patience          = 4           # number of epochs to wait before early stopping
        self.monitor           = "val_loss"  # what to track for EarlyStopping
        self.mode              = "min"       # stop when no-longer decreasing

        # Hardware parameters
        self.accelerator       = 'auto'      # device to use for training
        self.strategy          = 'auto'      # high-level how to do training
        self.devices           = 'auto'      # Device to use for model
        self.num_nodes         = 1           # number of gpu nodes to use
        self.precision         = '32'        # 32-bit floating point
        self.gradient_clip_val = 10.         # max abs. value for gradient
        self.logger            = True        # default Tensorboard logging

        # Checkpointing parameters
        self.every_n_epochs    = 5           # Number of epochs to go before checkpointing
        self.save_top_k        = 10          # Number of best checkpoint models to save

        # Safety parameters
        self.max_epochs        = 500         # max number of epochs to run

        self.process_config(config)

    def get_log_path(self, fold: int):
        return os.path.join(self.output_dirpath, "lightning_logs", f"version_{fold}")

    def get_ckpt_path(self, fold: int):
        return glob.glob(os.path.join(self.get_log_path(fold), "checkpoints", "*.ckpt"))[-1]

    def optimize_params(self, fold: int, static_params: dict={}):
        pass
