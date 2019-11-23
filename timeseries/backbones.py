from collections import namedtuple

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from .dataload import TimeDataset


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
        return torch.stack(outs, dim=1)[:, -1], h_state


class PredictNet(nn.Module):
    def __init__(self, net_path=None, **kwargs):
        super(PredictNet, self).__init__()
        self.cfg = self.parse_args(**kwargs)

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        # self.device = torch.device('cpu')

        self.net = Rnn(
            input_size=self.cfg.input_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=self.cfg.batch_first,
            out_scale=self.cfg.out_scale
        )

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'input_size': 17,
            'hidden_size': 32,
            'num_layers': 1,
            'batch_first': True,
            'out_scale': 2,
            'learning_rate': 0.01,
            'time_step': 17,
            'batch_size': 1,
            'epoch_num': 2000}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.enable_grad()
    def train(self, save_dir='pretrained'):
        self.net.train()

        h_state = None
        losses = []
        count = 0

        timeDataset = TimeDataset()
        train_loader = DataLoader(dataset=timeDataset,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=False)
        # print(len(train_loader))
        for i, data in enumerate(train_loader):
            if count == self.cfg.epoch_num:
                break
            count += 1
            inputs, labels = data
            x = inputs.to(self.device)
            y = labels.to(self.device)
            prediction, h_state = self.net(x, h_state)
            h_state = h_state.detach()
            # print(prediction)

            loss = self.loss_func(prediction, y)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # timeDataset = TimeDataset()
        # train_loader = DataLoader(dataset=timeDataset,
        #                           batch_size=self.cfg.batch_size,
        #                           shuffle=False)
        # count = 0
        # for j in range(self.cfg.epoch_num // len(train_loader) + 1):
        #     for i, data in enumerate(train_loader):
        #         if (j * len(train_loader) + i - count == self.cfg.epoch_num):
        #             break
        #         inputs, labels = data
        #         print(labels.shape)
        #         if (labels.shape[0] != self.cfg.batch_size):
        #             count += 1
        #             continue
        #
        #         x = inputs.to(self.device)
        #         y = labels.to(self.device)
        #         prediction, h_state = self.net(x, h_state)
        #         h_state = h_state.detach()
        #
        #         loss = self.loss_func(prediction, y)
        #         losses.append(loss.item())
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        # plt.plot(steps, y_np.flatten(), 'r-')
        # plt.plot(steps, prediction.detach().cpu().numpy().flatten(), 'b-')
        # plt.show()
        plt.plot(range(self.cfg.epoch_num), losses, 'b-')
        plt.show()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net_path = os.path.join(
            save_dir, 'predictnet_e%d.pth' % (self.cfg.epoch_num + 0))
        torch.save(self.net.state_dict(), net_path)

    @torch.no_grad()
    def test(self, seqs, state=None, flag=1):
        self.net.eval()
        # start, end = seqs * np.pi, (seqs + 1) * np.pi
        # steps = np.linspace(start, end, self.cfg.time_step, dtype=np.float32)
        # x_np = np.sin(steps)
        # y_np = np.cos(steps)
        # x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).to(self.device)
        # y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).to(self.device)
        # h_state = state
        # prediction, h_state = self.net(x, h_state)
        # plt.plot(steps, y_np.flatten(), 'r-')
        # plt.plot(steps, prediction.detach().cpu().numpy().flatten(), 'b-')
        # plt.show()
        x = torch.from_numpy(seqs).to(self.device)
        h_state = state
        prediction, h_state = self.net(x, h_state)
        if (flag == 0):
            return prediction[0][:],h_state
        else:
            return prediction[0][:],h_state
