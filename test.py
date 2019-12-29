import numpy as np
import os
import torch
# from siamfc.GIoUloss import GiouLoss

# a = np.array([[[1,2],[3,4]]])
# a = torch.from_numpy(a)
# print(a,a.transpose(-2,-1))
# a = np.random.randn(3,4)
# print(a,a.shape)
#
# b = a[None,:]
# print(b,b.shape)
#
# c = np.expand_dims(a,axis=0)
# print(b.shape)
#
# d = (a>0.5).astype(np.int32)
# e = (a<=0.5).astype(np.int32)
#
# print(d,e)
# x = d.sum()
# y = e.sum()
# print(x,y)

# coding = utf-8
import torch


def Giou_loss(box_pre, box_gt, method='mean'):
    '''
    input:
    box_pre: shape(N, 4)
    predicted
    x1, y1, x2, y2
    box_gt: shape(N, 4)
    groundtruth
    x1, y1, x2, y2
    return:
    Giou
    loss
    '''
    assert box_pre.shape == box_gt.shape

    # 并集区域坐标
    xx1 = torch.max(box_pre[:, 0], box_gt[:, 0])
    yy1 = torch.max(box_pre[:, 1], box_gt[:, 1])
    xx2 = torch.min(box_pre[:, 2], box_gt[:, 2])
    yy2 = torch.min(box_pre[:, 3], box_gt[:, 3])

    # 预测坐标面积
    box_pre_area = (box_pre[:, 2] - box_pre[:, 0] + 1) * (box_pre[:, 3] - box_pre[:, 1] + 1)
    # 标签坐标面积
    box_gt_area = (box_gt[:, 2] - box_gt[:, 0] + 1) * (box_gt[:, 3] - box_gt[:, 1] + 1)
    # inter 面积
    inter = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
    union = box_pre_area + box_gt_area - inter
    iou = inter / union

    # 最小封闭形状坐标
    xx1_c = torch.min(box_pre[:, 0], box_gt[:, 0])
    yy1_c = torch.min(box_pre[:, 1], box_gt[:, 1])
    xx2_c = torch.max(box_pre[:, 2], box_gt[:, 2])
    yy2_c = torch.max(box_pre[:, 3], box_gt[:, 3])

    # C面积
    area_c = (xx2_c - xx1_c) * (yy2_c - yy1_c)

    # Giou
    giou = iou - (area_c - union) / area_c

    giou_loss = 1 - giou

    if (method == 'mean'):
        return giou_loss.mean()
    else:
        return giou_loss.sum()

x = torch.Tensor([[149.40,535.86,301.53,406.86]])
y = torch.Tensor([[208,509,302,357]])
cri = Giou_loss(x,y)
print(cri)

# x = torch.randn(2, 3, 3)
# print(x)
# x= x.flip(-1)
# print(x)
# x = x.permute(0, 2, 1)
# print(x)


# print(np.hanning(10))
# print(np.outer(np.hanning(10),np.hanning(10)))

# c = 0.5 * np.sum([100,200])
# print(np.prod(np.array([100.,200.],dtype=np.float32)+150))

# a = np.random.randn(3,2,4)
# print(np.stack(a, axis=0)[:,-1].shape)
#
# b = np.array([[[1,2,3],[4,5,6],[1,2,3]],[[1,2,3],[4,5,6],[1,2,3]]])
# print(np.argmax(np.amax(b,axis=(1,2))))

# x = np.arange(4) - (4 - 1) / 2
# y = np.arange(4) - (4 - 1) / 2
# print(x)
# x, y = np.meshgrid(x, y)
# print(np.meshgrid(x, y))

# def listdir(path, list_name):
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             listdir(file_path, list_name)
#         elif os.path.splitext(file_path)[1]=='.jpeg':
#             list_name.append(file_path)
#
# print(listdir('H:/datasets/OTB100/BlurBody'))

# for i in os.listdir('H:/datasets/OTB100'):
    # if os.path.isfile(i):
    #     print(os.path.join(os.getcwd(),i))
# with open('names.txt','w+') as f:
#     for i in os.listdir('H:/datasets/OTB100'):
#     # f.read()
#         f.write(i+'\n')


# a = np.load("tools/feature_num.npy")
# b = 0
# for i in range(a.shape[0]):
#     b += a[i]
# print(a)

# from PIL import Image
# from got10k.datasets import GOT10k
# from got10k.utils.viz import show_frame
#
# dataset = GOT10k(root_dir='H:/datasets/GOT-10k', subset='train', return_meta=True)

# print(dataset)
# indices = np.random.permutation(len(dataset))
# print(len(indices))
# indexing
# img_file, anno, meta = dataset[10]
# print(anno[0])

# print(len(img_file), len(anno))
# print(anno[0])

# for-loop
# for s, (img_files, anno) in enumerate(dataset):
#     seq_name = dataset.seq_names[s]
#     print('Sequence:', seq_name)
#
#     # show all frames
#     for f, img_file in enumerate(img_files):
#         image = Image.open(img_file)
#         show_frame(image, anno[f, :])
# a = np.array([[[1,1],[2,2]],[[2,2],[3,3]]])
# print(np.where(a>1))
# b = 1-(a==1)
# print(b)
# print(a[b])
# a = a[::-1]
# print(a)



