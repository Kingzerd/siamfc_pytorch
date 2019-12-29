from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        self.conv_11 = nn.Conv2d(256,1,1,1)
        self.update = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)

        concat = []
        for i in range(nz):
            tmp = F.conv2d(x[i].unsqueeze(0), z[i].unsqueeze(1), groups=x.size(1))
            tmp = self.conv_11(tmp)
            concat.append(tmp)

        concat = torch.cat(concat,0)
        # print(concat.shape)
        return concat


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
