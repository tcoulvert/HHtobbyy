# ML packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class InclusiveNetwork(nn.Module):
    def __init__(self, num_hiddens=2, initial_node=500, dropout=0.5, gru_layers=2, gru_size=50, dropout_g=0.1, rnn_input=6, dnn_input=21):
        super(InclusiveNetwork, self).__init__()
        self.dropout = dropout
        self.dropout_g = dropout_g
        self.hiddens = nn.ModuleList()
        nodes = [initial_node]
        for i in range(num_hiddens):
            nodes.append(int(nodes[i]/2))
            self.hiddens.append(nn.Linear(nodes[i],nodes[i+1]))
        # self.gru = nn.GRU(input_size=7, hidden_size=gru_size, num_layers=gru_layers, batch_first=True, dropout=self.dropout_g)
        self.gru = nn.GRU(input_size=rnn_input, hidden_size=gru_size, num_layers=gru_layers, batch_first=True, dropout=self.dropout_g)
        self.merge = nn.Linear(dnn_input+gru_size,initial_node)
        self.out = nn.Linear(nodes[-1],2)

    def forward(self, particles, hlf):
        _, hgru = self.gru(particles)
        hgru = hgru[-1] # Get the last hidden layer
        x = torch.cat((hlf,hgru), dim=1)
        x = F.dropout(self.merge(x), training=self.training, p=self.dropout)
        for i in range(len(self.hiddens)):
            x = F.relu(self.hiddens[i](x))
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.out(x)
        return F.log_softmax(x, dim=1)