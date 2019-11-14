from __future__ import absolute_import

from timeseries import PredictNet
import numpy as np

if __name__ == '__main__':
    # root_dir = os.path.abspath('~/data/GOT-10k')
    net_path = 'pretrained/predictnet_e500.pth'

    predictnet = PredictNet(net_path=net_path)
    seqs = np.random.randint(100, 400,[1,5,4]).astype(np.float32)
    print(seqs)
    result = predictnet.test(seqs,flag=0)
    print(result)
