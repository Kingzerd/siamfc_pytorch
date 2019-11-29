import torch
from torch import nn
import numpy as np


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, out_scale):
        super(Rnn, self).__init__()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

        self.out = nn.Linear(hidden_size, out_scale)

    def forward(self, x, h_state=None):
        h = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        r_out, h_state = self.rnn(x, h)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1)[:, -1]