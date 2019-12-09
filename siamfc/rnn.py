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
        self.rnn1 = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.rnn2 = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.rnn3 = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        self.conv1 = nn.Conv2d(4, 1, 1, 1)
        self.conv2 = nn.Conv2d(4, 4, 17, 1)
        self.relu1 = nn.ReLU(inplace=True)

        self.out = nn.Linear(hidden_size, out_scale)

    def forward(self, x, h_state=None):
        x1, x2, x3 = self.four_copy(x)

        h1 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        h2 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        h3 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        h = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        # c = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        r_out, h_state = self.rnn(x, h)
        r_out1, h_state1 = self.rnn1(x1, h1)
        r_out2, h_state2 = self.rnn2(x2, h2)
        r_out3, h_state3 = self.rnn3(x3, h3)
        x = self.conv1(torch.stack((r_out, r_out1, r_out2, r_out3), -1).permute(0, 3, 1, 2)).squeeze(1)

        x1, x2, x3 = self.four_copy(x)
        r_out, h_state = self.rnn(x, h)
        r_out1, h_state1 = self.rnn1(x1, h1)
        r_out2, h_state2 = self.rnn2(x2, h2)
        r_out3, h_state3 = self.rnn3(x3, h3)
        x = self.conv2(torch.stack((r_out, r_out1, r_out2, r_out3), -1).permute(0, 3, 1, 2)).squeeze(2).squeeze(2)
        # x = self.relu1(x)

        # print(x, x.shape)
        # outs = []
        # for time in range(x.size(1)):
        #     outs.append(self.out(x[:, time, :]))
        # return torch.stack(outs, dim=1)[:, -1]
        return x

    def four_copy(self, x):
        batch = x.size(0)
        x1 = x.clone().cpu().detach().numpy()
        for i in range(batch):
            x1[i] = x1[i][::-1]
        x1 = torch.from_numpy(x1)
        x2 = x.transpose(-2, -1).contiguous()
        x3 = x2.clone().cpu().detach().numpy()
        for i in range(batch):
            x3[i] = x3[i][::-1]
        x3 = torch.from_numpy(x3)
        x1 = x1.cuda()
        x2 = x2.cuda()
        x3 = x3.cuda()
        return x1, x2, x3
