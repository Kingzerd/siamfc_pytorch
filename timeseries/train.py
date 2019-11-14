from __future__ import absolute_import

from timeseries import PredictNet

if __name__ == '__main__':
    # root_dir = os.path.abspath('~/data/GOT-10k')
    predictnet = PredictNet()
    predictnet.train()
