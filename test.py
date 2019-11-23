import numpy as np
import os
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
# print(np.stack(a, axis=0).shape)
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
#     # if os.path.isfile(i):
#         print(os.path.join(os.getcwd(),i))

a = np.load("tools/feature_num.npy")
# b = 0
# for i in range(a.shape[0]):
#     b += a[i]
print(a)
