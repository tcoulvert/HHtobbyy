from torch import optim, nn
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, num_nodes))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(act_fn())
            layers.append(nn.Dropout(p=dropout_prob))

        # Output layer
        layers.append(nn.Linear(num_nodes, output_size))
        #layers.append(nn.Softmax(dim=1))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, weights = batch

        logits = self(x)
        
        loss = nn.functional.cross_entropy(logits, y, reduction='none')
        weighted_loss = (weights * loss).sum() / weights.sum()

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", weighted_loss)
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, weights = batch

        logits = self(x)
        
        loss = nn.functional.cross_entropy(logits, y, reduction='none')
        weighted_loss = (weights * loss).sum() / weights.sum()

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", weighted_loss)
        return weighted_loss
    
    def test_step(self, batch, batch_idx):
        x, y, weights = batch

        logits = self(x)
        
        loss = nn.functional.cross_entropy(logits, y, reduction='none')
        weighted_loss = (weights * loss).sum() / weights.sum()

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", weighted_loss)
        return weighted_loss
