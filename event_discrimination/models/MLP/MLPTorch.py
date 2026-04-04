from torch import optim, nn
import pytorch_lightning as pl


class MLPTorch(pl.LightningModule):
    def __init__(self, input_size, num_layers, num_nodes, output_size, dropout_prob, class_weights=None):
        super(MLPTorch, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, num_nodes))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_prob))

        # Output layer
        layers.append(nn.Linear(num_nodes, output_size))
        layers.append(nn.LogSoftmax(dim=-1))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

        # Multiclass loss
        self.multi_loss = nn.NLLLoss(weight=class_weights, reduction='none')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-5, weight_decay=5e-5)
        return optimizer

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
    
    def test_step(self, batch, batch_idx):
        x, y, weights = batch

        logits = self(x)
        weighted_loss = self.compute_loss(logits, y, weights)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", weighted_loss)
        return weighted_loss
