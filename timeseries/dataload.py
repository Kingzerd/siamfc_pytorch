from __future__ import absolute_import

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, seq_dir='H:/datasets/OTB100/BlurBody'):
        self.seq_dir = seq_dir
        self.img_files = sorted(glob.glob(self.seq_dir + '/img/*.jpg'))
        self.anno = np.loadtxt(self.seq_dir + '/groundtruth_rect.txt', delimiter=',', dtype=np.float32)
        # print(anno, anno.shape)

        self.total_len = self.anno.shape[0]
        self.batch_len = 5
        self.input_size = 4
        self.stride = 1
        self.seq_len = int((self.total_len - self.batch_len) / self.stride)

        self.x = []
        self.y = []
        for i in range(self.seq_len):
            content_1 = []
            content_2 = []
            for j in range(self.batch_len):
                content_1.append(self.anno[i + j])
                content_2.append([self.anno[i + j+1][2]/self.anno[i + j][2],self.anno[i + j+1][3]/self.anno[i + j][3]])
                # if (j == self.batch_len - 1):
                #     self.y.append([self.anno[i + j + 1][2], self.anno[i + j + 1][3]])
            self.x.append(content_1)
            self.y.append(content_2)

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        # print(self.y.shape)
        self.x_data = torch.from_numpy(self.x)
        self.y_data = torch.from_numpy(self.y)
        self.len = self.seq_len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.seq_len


if __name__ == '__main__':
    # with open('H:/datasets/OTB100/Woman/groundtruth_rect.1.txt') as f:
    #     with open('H:/datasets/OTB100/Woman/groundtruth_rect.txt',mode="w", encoding="utf-8") as f1:
    #         f1.write(f.read().replace('\t',','))

    path = 'H:/datasets/OTB100/'
    for i in os.listdir(path):
        cur = path+i
        print(cur)
        timeDataset = TimeDataset(cur)

        train_loader = DataLoader(dataset=timeDataset,
                                  batch_size=5,
                                  shuffle=False)
        print(len(train_loader))

        # for i, data in enumerate(train_loader):
        #     inputs, labels = data
        #     print(labels.shape)

    b = np.load('D:/PycharmProjects/siamfc-pytorch-master/tools/feature.npy')
    print(b, b.shape)
