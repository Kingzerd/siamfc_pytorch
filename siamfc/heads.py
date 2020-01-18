from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']

class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        # self.norm = nn.BatchNorm2d(256)
        self.conv_1 = nn.Conv2d(256, 1, 1, 1)
        self.conv_11 = nn.Sequential(
            nn.Conv2d(256,256,6,1,1),
            nn.Conv2d(256,1,5,1,1),
            nn.BatchNorm2d(1),
            # # _BatchNorm2d(1),
            # nn.ReLU(inplace=False),
            nn.Tanh()
        )

    
    def forward(self, z, x, frnn):
        return self._fast_xcorr(z, x, frnn)
    
    def _fast_xcorr(self, z, x, frnn):
        # fast cross correlation
        nz = z.size(0)
        # group = z.size(1)
        concat = []

        # b, c, h, w = x.size()
        # x = x.view(-1, nz*c,h,w)
        # print(F.conv2d(x, z, groups=group).size())  # 1,8*256,17,17
        # print(F.conv2d(x, z.permute(1, 0, 2, 3), groups=32).size())  # 8,256,17,17

        for i in range(nz):
            tmp = F.conv2d(x[i].unsqueeze(0), z[i].unsqueeze(1), groups=x.size(1))
            concat.append(tmp)
        concat = torch.cat(concat, 0)

        # concat = F.conv2d(x, z.permute(1, 0, 2, 3), groups=x.size(1)/x.size(0))
        # concat = self.norm(concat)
        score = self.conv_1(concat)
        # next_rnn = self.conv_11(concat)
        next_rnn = self.conv_11(frnn)
        # print(concat.shape, next_rnn.shape)
        # return next_rnn * self.out_scale, score * self.out_scale
        return next_rnn, score


# class SiamFC(nn.Module):
#
#     def __init__(self, out_scale=0.001):
#         super(SiamFC, self).__init__()
#         self.out_scale = out_scale
#
#     def forward(self, z, x):
#         return self._fast_xcorr(z, x) * self.out_scale
#
#     def _fast_xcorr(self, z, x):
#         # fast cross correlation
#         nz = z.size(0)
#         nx, c, h, w = x.size()
#         x = x.view(-1, nz * c, h, w)
#         out = F.conv2d(x, z, groups=nz)
#         out = out.view(nx, -1, out.size(-2), out.size(-1))
#         return out
