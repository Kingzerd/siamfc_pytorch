import torch.nn as nn


class UpdateNet(nn.Module):
    def __init__(self, config=None):
        super(UpdateNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        # t = torch.cat((x, y, z), 0)
        response = self.update(x)
        return response
