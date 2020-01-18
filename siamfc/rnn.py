import torch
from torch import nn
import numpy as np


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

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
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 1, 1, 1, bias=True),
            # nn.BatchNorm2d(1),
            # _BatchNorm2d(1),
            # nn.Tanh()
            # nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 4, 1, 1, bias=True),
            # nn.BatchNorm2d(4),
            # _BatchNorm2d(4),
            # nn.Tanh()
            # nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(4, 4, 17, 1, bias=True)

        self.out = nn.Linear(hidden_size, out_scale)
        self.h = None
        self.h1 = None
        self.h2 = None
        self.h3 = None

    def forward(self, x, h_state=None):
        x1, x2, x3 = self.four_copy(x)
        if self.h is None:
            self.h1 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
            self.h2 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
            self.h3 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
            self.h = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        # c = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        r_out, self.h = self.rnn(x, self.h)
        r_out1, self.h1 = self.rnn1(x1, self.h1)
        r_out2, self.h2 = self.rnn2(x2, self.h2)
        r_out3, self.h3 = self.rnn3(x3, self.h3)
        x = self.conv1(torch.stack((r_out, r_out1, r_out2, r_out3), -1).permute(0, 3, 1, 2)).squeeze(1)

        x1, x2, x3 = self.four_copy(x)
        r_out, self.h = self.rnn(x, self.h)
        r_out1, self.h1 = self.rnn1(x1, self.h1)
        r_out2, self.h2 = self.rnn2(x2, self.h2)
        r_out3, self.h3 = self.rnn3(x3, self.h3)
        # x = self.conv3(self.conv2(torch.stack((r_out, r_out1, r_out2, r_out3), -1).permute(0, 3, 1, 2))).squeeze(2).squeeze(2)
        x = self.conv2(torch.stack((r_out, r_out1, r_out2, r_out3), -1).permute(0, 3, 1, 2))
        # x = self.conv3(self.conv2(torch.stack((x, x1, x2, x3), -1).permute(0, 3, 1, 2))).squeeze(
        #     2).squeeze(2)

        self.h = self.h.clone().detach()
        self.h1 = self.h1.clone().detach()
        self.h2 = self.h2.clone().detach()
        self.h3 = self.h3.clone().detach()
        # print(x.shape)
        # outs = []
        # for time in range(x.size(1)):
        #     outs.append(self.out(x[:, time, :]))
        # return torch.stack(outs, dim=1)[:, -1]
        return x.permute(0,2,3,1)

    def four_copy(self, x):
        x1 = x.clone().flip(-1)
        x2 = x.clone().permute(0,2,1)
        x3 = x2.clone().flip(-1)
        return x1, x2, x3
