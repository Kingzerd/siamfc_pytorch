from __future__ import absolute_import

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, seq_dir='H:/datasets/OTB100/'):
        self.feature = np.load('D:/PycharmProjects/siamfc-pytorch-master/tools/feature.npy')
        self.feature_num = np.load('D:/PycharmProjects/siamfc-pytorch-master/tools/feature_num.npy')
        self.x = []
        self.y = []
        self.dataset_count = -1
        self.feature_count = 0
        path = seq_dir
        count = 0
        for j in os.listdir(path):
            if self.dataset_count > -1:
                self.feature_count += self.feature_num[self.dataset_count]
                # print(self.feature_count)
            self.dataset_count += 1

            self.seq_dir = path + j
            self.img_files = sorted(glob.glob(self.seq_dir + '/img/*.jpg'))
            self.anno = np.loadtxt(self.seq_dir + '/groundtruth_rect.txt', delimiter=',', dtype=np.float32)

            count += self.anno.shape[0]

            self.total_len = self.anno.shape[0]
            self.batch_len = 1
            self.stride = 1
            self.seq_len = int((self.total_len - self.batch_len-1) / self.stride)

            for i in range(self.seq_len):
                content_1 = []
                content_2 = []
                for j in range(self.batch_len):
                    # print(self.feature_count+i+j)
                    content_1.append(self.feature[self.feature_count+i+j])
                    content_2.append([self.anno[i + j+2][2]/self.anno[i + j+1][2],self.anno[i + j+2][3]/self.anno[i + j +1][3]])
                    # if (j == self.batch_len - 1):
                    #     self.y.append([self.anno[i + j + 1][2], self.anno[i + j + 1][3]])
                self.x.append(content_1)
                self.y.append(content_2)

        self.x = np.array(self.x).squeeze()
        self.y = np.array(self.y).squeeze()
        self.len = self.x.shape[0]
        # print(self.len)

        self.x_data = torch.from_numpy(self.x)
        self.y_data = torch.from_numpy(self.y)
        # self.len = self.seq_len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # with open('H:/datasets/OTB100/Woman/groundtruth_rect.1.txt') as f:
    #     with open('H:/datasets/OTB100/Woman/groundtruth_rect.txt',mode="w", encoding="utf-8") as f1:
    #         f1.write(f.read().replace('\t',','))

    timeDataset = TimeDataset()

    train_loader = DataLoader(dataset=timeDataset,
                              batch_size=1,
                              shuffle=False)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # print(inputs.shape, labels.shape)
    # print(len(train_loader))
