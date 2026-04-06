# ML packages
import gpustat

# Workspace packages
from HHtobbyy.event_discrimination.DFDataset import DFDataset
from HHtobbyy.event_discrimination.models import ModelConfig

################################


class MLPConfig(ModelConfig):
    def __init__(self, dfdataset: DFDataset, config: dict):
        self.dfdataset = dfdataset

        # Architecture parameters
        self.num_layers        = 5           # number of hidden layers
        self.num_nodes         = 1024        # dimensionality of hidden layers
        self.dropout_prob      = 0.25        # probability of dropping connections

        # Eary stopping parameters
        self.min_delta         = 0.          # smallest val_loss difference
        self.patience          = 4           # number of epochs to wait before early stopping
        self.monitor           = "val_loss"  # what to track for EarlyStopping
        self.mode              = "min"       # stop when no-longer decreasing

        # Hardware parameters
        try:
            gpustat.print_gpustat()
            self.accelerator   = 'gpu'       # device to use for training
        except:
            self.accelerator   = 'cpu'
        self.strategy          = 'auto'      # high-level how to do training
        self.num_nodes         = 1           # number of gpu nodes to use
        self.precision         = '32'        # 32-bit floating point
        self.gradient_clip_val = 10.         # max abs. value for gradient
        self.logger            = True        # default Tensorboard logging

        # Safety parameters
        self.max_epochs        = 500         # max number of epochs to run

        super().process_config(config)

    def optimize_params(self, fold: int, static_params: dict={}):
        pass
