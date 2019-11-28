import numpy as np
import os
import torch

a = np.array([1,2,3,4])
a = torch.from_numpy(a)
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


