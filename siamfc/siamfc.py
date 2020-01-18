from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .rnn import Rnn
from .updatenet import UpdateNet
from .losses import BalancedLoss, Regressloss
from .datasets import Pair
from .transforms import SiamFCTransforms
from . import GIoUloss
# from timeseries import PredictNet

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head, update, regression):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
        self.update = update
        self.regression = regression

    def forward(self, z, x, z_copy, template=None):
        z = self.backbone(z)
        x = self.backbone(x)
        z_copy = self.backbone(z_copy)
        if template is None:
            template = z.clone()
        tmp = torch.cat([z, template.detach()], 1)
        new_template = self.update(tmp)
        # next_rnn, concat = self.head(new_template, x)
        next_rnn, score = self.head(z, x, x-z_copy)

        dxywh = self.regression(next_rnn.squeeze(1))
        return score, dxywh, new_template
        # return score, dxywh, new_template
        # return score, dxywh, template


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, keep_train=False, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        # self.net = Net(
        #     backbone=AlexNetV1(),
        #     head=SiamFC(self.cfg.feature_scale))
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.feature_scale),
            update=UpdateNet(),
            regression=Rnn(
                input_size=self.cfg.input_size,
                hidden_size=self.cfg.hidden_size,
                num_layers=self.cfg.num_layers,
                batch_first=self.cfg.batch_first,
                out_scale=self.cfg.out_scale
            ))
        ops.init_weights(self.net)
        # self.h_state = None
        self.template = None

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.gussian_loss = Regressloss()
        self.giou_loss = GIoUloss.GiouLoss()

        # setup optimizer
        # self.optimizer = optim.SGD(
        #     self.net.parameters(),
        #     lr=self.cfg.initial_lr,
        #     weight_decay=self.cfg.weight_decay,
        #     momentum=self.cfg.momentum)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay)

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
        self.feature = []
        self.feature_num = []
        if keep_train:
            self.net.load_state_dict(torch.load('pretrained/siamfc_alexnet_eeee20.pth'))

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'feature_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,  #0.176
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 20,
            'batch_size': 8,
            'num_workers': 16,
            'initial_lr': 1e-3,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
            'input_size': 17,
            'hidden_size': 17,
            'num_layers': 1,
            'batch_first': True,
            'out_scale': 4,
            'time_step': 17}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # self.history = []
        # self.h_state = None
        # self.feature_count = 0
        # for i in range(self.cfg.time_step):
        #     self.history.append(box)
        # net_path = '../timeseries/pretrained/predictnet_e2000.pth'
        # self.predictnet = PredictNet(net_path=net_path)

        # # convert box to 0-indexed and center based [y, x, h, w]
        # box = np.array([
        #     box[1] - 1 + (box[3] - 1) / 2,
        #     box[0] - 1 + (box[2] - 1) / 2,
        #     box[3], box[2]], dtype=np.float32)
        # self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        self.update_template(img, box, 'first')
        # # exemplar and search sizes
        # context = self.cfg.context * np.sum(self.target_sz)
        # self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        # self.x_sz = self.z_sz * \
        #             self.cfg.instance_sz / self.cfg.exemplar_sz
        #
        # # exemplar image
        # self.avg_color = np.mean(img, axis=(0, 1))
        # z = ops.crop_and_resize(
        #     img, self.center, self.z_sz,
        #     out_size=self.cfg.exemplar_sz,
        #     border_value=self.avg_color)
        #
        # # exemplar features
        # z = torch.from_numpy(z).to(
        #     self.device).permute(2, 0, 1).unsqueeze(0).float()
        # self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update_template(self, img, box, flag='then'):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        if flag == 'first':
            self.center, self.target_sz = box[:2], box[2:]
        center, target_sz = box[:2], box[2:]

        # exemplar and search sizes
        context = self.cfg.context * np.sum(target_sz)
        if flag == 'first':
            self.z_sz = np.sqrt(np.prod(target_sz + context))
            self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, center, z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        z_copy = ops.crop_and_resize(
            img, center, x_sz,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color)
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        z_copy = torch.from_numpy(z_copy).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        if flag == 'first':
            self.initial_kernel = self.net.backbone(z)
            self.initial_frnn = self.net.backbone(z_copy)
        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        crops = x[:]
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        x = self.net.backbone(x)

        if self.template is None:
            self.template = self.kernel.clone()
        templa = self.net.update(torch.cat([self.kernel, self.template], 1))
        # forrnn, concat = self.net.head(templa, x)
        forrnn, responses = self.net.head(self.initial_kernel.repeat(len(self.scale_factors),1,1,1), x, x-self.initial_frnn)
        self.template = templa.clone().detach()
        responses = responses.squeeze(1).cpu().numpy()

        # r = responses.copy()
        # index = -1
        # for response in responses:
        #     index += 1
        #     for i in range(response.shape[0]):
        #         for j in range(response.shape[1]):
        #             r[index][i][j] = max(response[i]) + max(response.T[j])
        # responses = r
        # print(responses.shape)

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        # response /= response.sum() + 1e-16
        response /= (response.max()-response.min())
        non_hann = response.copy()
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window

        forrnn = forrnn[scale_id]
        # initial -= initial.min()
        # initial /= initial.sum() + 1e-16
        # initial /= (initial.max() - initial.min())
        # print(initial)

        # print(initial.shape)
        dxywh = self.net.regression(forrnn)
        dxywh = dxywh.detach().cpu().numpy()
        dxywh = dxywh[0]
        # self.feature.append(initial)
        # self.feature_count += 1

        loc = np.unravel_index(response.argmax(), response.shape)
        dxywh_loc = np.array(loc)
        dxywh = dxywh[int(round(dxywh_loc[0]/16.0)),int(round(dxywh_loc[1]/16.0))]

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * np.sqrt(np.exp(dxywh[2]) * np.exp(dxywh[3]))
        # scale = (1 - self.cfg.scale_lr) * 1.0 + \
        #         self.cfg.scale_lr * self.scale_factors[scale_id] * np.sqrt(np.exp(dxywh[2]) * np.exp(dxywh[3]))
        print('scale: ', scale)

        # print('regression: ', dxywh, np.exp(dxywh[2]), np.exp(dxywh[3]))
        dx = dxywh[0] * self.target_sz[1]
        dy = dxywh[1] * self.target_sz[0]
        print('regression: ', dx, dy, np.exp(dxywh[2]), np.exp(dxywh[3]))
        scale_factor = scale
        # self.target_sz *= scale_factor
        # self.target_sz[1] *= np.exp(dxywh[2])
        # self.target_sz[0] *= np.exp(dxywh[3])
        target_sz_copy = self.target_sz.copy()
        target_sz_copy[1] *= np.exp(dxywh[2])
        target_sz_copy[0] *= np.exp(dxywh[3])
        # self.z_sz *= (np.exp(dxywh[2])+np.exp(dxywh[3]))/2
        # self.x_sz *= (np.exp(dxywh[2])+np.exp(dxywh[3]))/2

        # return 1-indexed and left-top based bounding box
        box = [
            self.center[1] + dx + 1 - (target_sz_copy[1] - 1) / 2,
            self.center[0] + dy + 1 - (target_sz_copy[0] - 1) / 2,
            target_sz_copy[1], target_sz_copy[0]]
        # box = [
        #     self.center[1] + dy+1 - (target_sz_copy[1] - 1) / 2,
        #     self.center[0] + dx+1 - (target_sz_copy[0] - 1) / 2,
        #     int(self.originwh[0] * np.exp(dxywh[2])), int(self.originwh[1] * np.exp(dxywh[3]))]
        self.update_template(img, [box[1], box[0], box[3], box[2]])
        # box = [
        #     int(self.originxy[0]),
        #     int(self.originxy[1]),
        #     int(self.originwh[0]), int(self.originwh[1])]
        # self.originxy[0] += int(dx)
        # self.originxy[1] += int(dy)
        # self.originwh[0] *= np.exp(dxywh[2])
        # self.originwh[1] *= np.exp(dxywh[3])

        # self.history = self.history[1:]
        # self.history.append(box)
        box = np.array(box)
        box_copy = box.copy()
        fac1 = 255/640
        fac2 = 255/480
        box_copy[0] *= fac1
        box_copy[1] *= fac2
        box_copy[2] *= fac1
        box_copy[3] *= fac2
        print('box: ', box)
        if self.feature_flag==1:
            ops.print_feature_2(crops[scale_id], response, box_copy)
        if self.feature_flag==2:
            ops.print_feature_3(crops[scale_id], response, box_copy, non_hann)
        return box

    def write_feature(self):
        np.save('feature.npy', np.array(self.feature))
        np.save('feature_num.npy', np.array(self.feature_num))

    def track(self, img_files, box, visualize=False, rnn_flag=0, feature_flag=0):
        self.originxy = box[:2]
        self.originwh = box[2:]
        self.rnn_flag = rnn_flag
        self.feature_flag = feature_flag
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])
        # self.feature_num.append(self.feature_count)
        return boxes, times

    def train_step(self, batch, temp, backward=True):
        # set network mode
        self.net.train(backward)

        length = 1
        shift = 3
        length_repeat = (length*2+1)**2

        label_dxywh = []
        for k in range(self.cfg.batch_size):
            tmp = []
            tmp.append((batch[3][k][0]-batch[2][k][0])/batch[2][k][2])
            tmp.append((batch[3][k][1]-batch[2][k][1])/batch[2][k][3])
            tmp.append(np.log(batch[3][k][2]/batch[2][k][2]))
            tmp.append(np.log(batch[3][k][3]/batch[2][k][3]))
            for cou in range(length_repeat):
                label_dxywh.append(tmp)
            # print(np.exp(tmp[2]), np.exp(tmp[3]))
        label_dxywh = torch.from_numpy(np.array(label_dxywh)).float()

        label_xyxy = []
        for k in range(self.cfg.batch_size):
            tmp = []
            tmp.append(batch[3][k][0])
            tmp.append(batch[3][k][1])
            tmp.append(batch[3][k][0]+batch[3][k][2])
            tmp.append(batch[3][k][1]+batch[3][k][3])
            for cou in range(length_repeat):
                label_xyxy.append(tmp)
        label_xyxy = torch.from_numpy(np.array(label_xyxy)).float()
        # print('label ', label_dxywh[0])

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        z_copy = batch[4].to(self.device, non_blocking=self.cuda)
        label_xyxy = label_xyxy.to(self.device, non_blocking=self.cuda)
        label_dxywh = label_dxywh.to(self.device, non_blocking=self.cuda)
        # label_dxywh = (label_dxywh-torch.min(label_dxywh))/(torch.max(label_dxywh)-torch.min(label_dxywh))

        with torch.set_grad_enabled(backward):
            # inference
            # responses = self.net(z, x)
            responses, dxywh, template = self.net(z, x, z_copy, temp)
            # b, c, h, w = dxywh.size()

            flagg = 0
            for cou in range(self.cfg.batch_size):
                # print(batch[5][cou])
                if batch[5][cou] == 1:
                    compute = dxywh[cou,9-length-shift:9+length-shift+1,9-length-shift:9+length-shift+1,:]
                elif batch[5][cou] == 2:
                    compute = dxywh[cou,9-length-shift:9+length-shift+1,9-length:9+length+1,:]
                elif batch[5][cou] == 3:
                    compute = dxywh[cou,9-length-shift:9+length-shift+1,9+shift-length:9+length+shift+1,:]
                elif batch[5][cou] == 4:
                    compute = dxywh[cou,9-length:9+length+1,9-length-shift:9+length-shift+1,:]
                elif batch[5][cou] == 6:
                    compute = dxywh[cou,9-length:9+length+1,9+shift-length:9+length+shift+1,:]
                elif batch[5][cou] == 7:
                    compute = dxywh[cou,9+shift-length:9+length+shift+1,9-length-shift:9+length-shift+1,:]
                elif batch[5][cou] == 8:
                    compute = dxywh[cou,9+shift-length:9+length+shift+1,9-length:9+length+1,:]
                elif batch[5][cou] == 9:
                    compute = dxywh[cou,9+shift-length:9+length+shift+1,9+shift-length:9+length+shift+1,:]
                else:
                    compute = dxywh[cou,9-length:9+length+1,9-length:9+length+1,:]
                # print(compute.shape)
                compute = compute.clone().view(-1,4)
                if flagg == 0:
                    final_dxywh = compute
                    flagg += 1
                else:
                    final_dxywh = torch.cat((final_dxywh, compute), 0)
            # calculate loss
            labels = self._create_labels(responses.size(), batch[5])
            cls_loss = self.criterion(responses, labels)
            # dxywh = dxywh.detach().cpu().numpy()
            dxywh = final_dxywh
            xyxy = dxywh.clone()
            for k in range(dxywh.shape[0]):
                xyxy[k][0] = dxywh[k][0] * (label_xyxy[k][2]-label_xyxy[k][0]) + (label_xyxy[k][2]+label_xyxy[k][0])/2
                xyxy[k][1] = dxywh[k][1] * (label_xyxy[k][3]-label_xyxy[k][1]) + (label_xyxy[k][3]+label_xyxy[k][1])/2
                xyxy[k][2] = torch.exp(dxywh[k][2]) * (label_xyxy[k][2]-label_xyxy[k][0])
                xyxy[k][3] = torch.exp(dxywh[k][3]) * (label_xyxy[k][3]-label_xyxy[k][1])
                if xyxy[k][2] <=0:
                    xyxy[k][2] = label_xyxy[k][2]
                if xyxy[k][3] <=0:
                    dxywh[k][3] = label_xyxy[k][3]
            for i in range(xyxy.shape[0]):
                xyxy[i][0] -= xyxy[i][2]/2
                xyxy[i][1] -= xyxy[i][3]/2
                if xyxy[i][0] < 0:
                    xyxy[i][0] = 0
                if xyxy[i][1] < 0:
                    xyxy[i][1] = 0
                xyxy[i][2] += xyxy[i][0]
                xyxy[i][3] += xyxy[i][1]
            # dxywh = (dxywh - torch.min(dxywh)) / (torch.max(dxywh) - torch.min(dxywh))
            # xyxy = torch.autograd.Variable(torch.from_numpy(np.array(xyxy)).float(), requires_grad=True)

            reg_loss1 = self.reg_loss(dxywh, label_dxywh)
            # print(label_dxywh.shape)
            gussian_loss = self.gussian_loss(dxywh, label_dxywh)
            reg_loss2 = self.giou_loss(xyxy, label_xyxy)
            # print(reg_loss)
            print('cls_loss ', cls_loss, 'reg_loss1 ', reg_loss1, 'reg_loss2 ', reg_loss2, 'gussian_loss ', gussian_loss)
            # print('cls_loss ', cls_loss, 'reg_loss1 ', reg_loss1, 'gussian_loss ', gussian_loss)
            loss = reg_loss1 + reg_loss2 + cls_loss + gussian_loss
            # loss = reg_loss1+cls_loss+gussian_loss

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                # cls_loss.backward(retain_graph=True)
                # reg_loss1.backward(retain_graph=True)
                # reg_loss2.backward()
                loss.backward()
                self.optimizer.step()

        return loss.item(), template

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            template = None
            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss, template = self.train_step(batch, template, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if (epoch+1) % 5 == 0:
                net_path = os.path.join(
                    save_dir, 'siamfc_alexnet_eee%d.pth' % (epoch + 1))
                torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size, t):
        # skip if same sized labels already created
        # if hasattr(self, 'labels') and self.labels.size() == size:
        if hasattr(self, 'labels'):
            temp = []
            for i in t:
                temp.append(self.labels[i - 1])
            temp = torch.from_numpy(np.array(temp)).to(self.device).float()
            return temp

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        def move_up(matrix, dis=3):
            matrix = matrix.copy()
            tmp = matrix[dis:, :].copy()
            matrix[-dis:, :] = matrix[:dis, :]
            matrix[:-dis, :] = tmp
            return matrix

        def move_down(matrix, dis=3):
            matrix = matrix.copy()
            tmp = matrix[:-dis, :].copy()
            matrix[:dis, :] = matrix[-dis:, :]
            matrix[dis:, :] = tmp
            return matrix

        def move_left(matrix, dis=3):
            matrix = matrix.copy()
            tmp = matrix[:, dis:].copy()
            matrix[:, -dis:] = matrix[:, :dis]
            matrix[:, :-dis] = tmp
            return matrix

        def move_right(matrix, dis=3):
            matrix = matrix.copy()
            tmp = matrix[:, :-dis].copy()
            matrix[:, :dis] = matrix[:, -dis:]
            matrix[:, dis:] = tmp
            return matrix

        self.labels = [1,2,3,4,5,6,7,8,9]
        self.labels_copy = self.labels.copy()
        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        self.labels_copy[4] = labels
        self.labels_copy[3] = move_left(self.labels_copy[4])
        self.labels_copy[5] = move_right(self.labels_copy[4])
        self.labels_copy[1] = move_up(labels)
        self.labels_copy[0] = move_left(self.labels_copy[1])
        self.labels_copy[2] = move_right(self.labels_copy[1])
        self.labels_copy[7] = move_down(labels)
        self.labels_copy[6] = move_left(self.labels_copy[7])
        self.labels_copy[8] = move_right(self.labels_copy[7])

        # repeat to size
        # labels = labels.reshape((1, 1, h, w))
        # labels = np.tile(labels, (n, c, 1, 1))
        count = 0
        for i in self.labels_copy:
            i = i.reshape((1, h, w))
            i = np.tile(i, (c, 1, 1))
            self.labels[count] = i
            count += 1

        # convert to tensors
        # self.labels = torch.from_numpy(self.labels).to(self.device).float()
        temp = []
        for i in t:
            temp.append(self.labels[i-1])
        temp = torch.from_numpy(np.array(temp)).to(self.device).float()
        return temp
