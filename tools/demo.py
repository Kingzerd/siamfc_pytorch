from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_ee10.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # seq_dir = os.path.expanduser('~/data/OTB/Crossing')
    path = 'H:/datasets/OTB100/'
    for i in os.listdir(path):
        # seq_dir = path + i
        seq_dir = 'H:/datasets/OTB100/Bolt2'
        img_files = sorted(glob.glob(seq_dir + '/img/*.jpg'))
        anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt', delimiter=',')

        tracker.track(img_files, anno[0], visualize=True, rnn_flag=0, feature_flag=0)
    # tracker.write_feature()
