import numpy as np
import os
import torch
import cv2
print(round(130/16))

e = np.array([])
print(e)

def move_up(matrix, dis):
    matrix = matrix.copy()
    tmp = matrix[:,dis:].copy()
    matrix[:, -dis:] = matrix[:, :dis]
    matrix[:, :-dis] = tmp
    return matrix

response = np.array([[1,2,3,1],[3,5,4,2],[2,4,5,3]]).astype('uint8')
print(response)
res = move_up(response, 1)
print(res)

# r = response.copy()
# for i in range(response.shape[0]):
#     for j in range(response.shape[1]):
#         r[i][j] = max(response[i]) + max(response.T[j])
# print(r)
r = cv2.resize(response, (6,6), interpolation=cv2.INTER_CUBIC)
print(r)
#
# if batch[5][cou] == 1:
#     compute = dxywh[cou, :11, :11, :]
# elif batch[5][cou] == 2:
#     compute = dxywh[cou, :11, 3:14, :]
# elif batch[5][cou] == 3:
#     compute = dxywh[cou, :11, 6:, :]
# elif batch[5][cou] == 4:
#     compute = dxywh[cou, 3:14, :11, :]
# elif batch[5][cou] == 6:
#     compute = dxywh[cou, 3:14, 6:, :]
# elif batch[5][cou] == 7:
#     compute = dxywh[cou, 6:, :11, :]
# elif batch[5][cou] == 8:
#     compute = dxywh[cou, 6:, 3:14, :]
# elif batch[5][cou] == 9:
#     compute = dxywh[cou, 6:, 6:, :]
# else:
#     compute = dxywh[cou, 3:14, 3:14, :]

# def Giou_loss(box_pre, box_gt, method='mean'):
#     '''
#     input:
#     box_pre: shape(N, 4)
#     predicted
#     x1, y1, x2, y2
#     box_gt: shape(N, 4)
#     groundtruth
#     x1, y1, x2, y2
#     return:
#     Giou
#     loss
#     '''
#     assert box_pre.shape == box_gt.shape
#
#     # 并集区域坐标
#     xx1 = torch.max(box_pre[:, 0], box_gt[:, 0])
#     yy1 = torch.max(box_pre[:, 1], box_gt[:, 1])
#     xx2 = torch.min(box_pre[:, 2], box_gt[:, 2])
#     yy2 = torch.min(box_pre[:, 3], box_gt[:, 3])
#
#     # 预测坐标面积
#     box_pre_area = (box_pre[:, 2] - box_pre[:, 0] + 1) * (box_pre[:, 3] - box_pre[:, 1] + 1)
#     # 标签坐标面积
#     box_gt_area = (box_gt[:, 2] - box_gt[:, 0] + 1) * (box_gt[:, 3] - box_gt[:, 1] + 1)
#     # inter 面积
#     inter = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
#     union = box_pre_area + box_gt_area - inter
#     iou = inter / union
#
#     # 最小封闭形状坐标
#     xx1_c = torch.min(box_pre[:, 0], box_gt[:, 0])
#     yy1_c = torch.min(box_pre[:, 1], box_gt[:, 1])
#     xx2_c = torch.max(box_pre[:, 2], box_gt[:, 2])
#     yy2_c = torch.max(box_pre[:, 3], box_gt[:, 3])
#
#     # C面积
#     area_c = (xx2_c - xx1_c) * (yy2_c - yy1_c)
#
#     # Giou
#     giou = iou - (area_c - union) / area_c
#
#     giou_loss = 1 - giou
#
#     if (method == 'mean'):
#         return giou_loss.mean()
#     else:
#         return giou_loss.sum()
#
# x = torch.Tensor([[149.40,535.86,301.53,406.86]])
# y = torch.Tensor([[208,509,302,357]])
# cri = Giou_loss(x,y)
# print(cri)


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


# for-loop
# for s, (img_files, anno) in enumerate(dataset):
#     seq_name = dataset.seq_names[s]
#     print('Sequence:', seq_name)
#
#     # show all frames
#     for f, img_file in enumerate(img_files):
#         image = Image.open(img_file)
#         show_frame(image, anno[f, :])




