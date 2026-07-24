from torch import optim, nn, Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightning as L


class MLPTorch(L.LightningModule):
    def __init__(self, input_size, num_layers, hidden_dim, output_size, dropout_prob, activation_func, learning_rate, learning_rate_decay, weight_decay, max_epochs, n_batches, class_weights: Tensor=None, **kwargs):
        super(MLPTorch, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learning_rate_decay = learning_rate_decay
        self.max_epochs = max_epochs
        self.n_batches = n_batches
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(getattr(nn, activation_func)())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(getattr(nn, activation_func)())
            layers.append(nn.Dropout(p=dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

        # Multiclass loss
        self.multi_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

        # Store hyperparameters in checkpoint
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            epochs=self.max_epochs,
            steps_per_epoch=self.n_batches,
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy='cos',  # Cosine annealing after peak
            div_factor=25.0,  # initial_lr = max_lr/25 = 4e-06
            final_div_factor=1e4  # final_lr = initial_lr/1e4 = 4e-10
        )
        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     first_cycle_steps=None, 
        #     max_lr=self.learning_rate, 
        #     min_lr=self.learning_rate*self.lr_decay_factor, 
        #     warmup_steps=self.n_batches,
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def backward(self, loss):
        loss.backward()

    def compute_loss(self, logits, y, weights):
        loss = self.multi_loss(logits, y)
        weighted_loss = (weights * loss).sum() / weights.sum()
        return weighted_loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, weights = batch

        logits = self(x)
        weighted_loss = self.compute_loss(logits, y, weights)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", weighted_loss)
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, weights = batch

        logits = self(x)
        weighted_loss = self.compute_loss(logits, y, weights)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", weighted_loss)
        return weighted_loss
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch

        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def test_step(self, batch, batch_idx):
        x, y, weights = batch

        logits = self(x)
        weighted_loss = self.compute_loss(logits, y, weights)

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", weighted_loss)
        return weighted_loss
