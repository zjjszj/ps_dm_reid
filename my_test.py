import torch
import random



#读入图片
# x=torch.zeros((2,3,2,2))
# x[:,:,1:2,0:3]=1
# print(x)
# #用均值代替
# c,h, w = x.size()[-3:]
# rh = round(0.5 * h)
# rw = round(0.5* w)
# sx = random.randint(0, h-rh)
# sy = random.randint(0, w-rw)
# for i in range(c):
#     scope=x[:,i,sx:sx+rh, sy:sy+rw]
#     scope_aveg=torch.sum(scope)/(rh*rw)
#     x[:, i, sx:sx + rh, sy:sy + rw]=scope_aveg
#
# print('x===',x)
class Test:

    def set_attr(self):
        self.a=2

from torchvision.models.resnet import Bottleneck, resnet50
import torch
import numpy as np
from datasets.psdb import psdb
import os.path as osp
import sys

if __name__ == '__main__':
    # det=np.array([[1,2,3,4,5],[1,2,3,4,5]])
    # gallery_feat=np.array([[0.1,0.2,0.3,0.4,0.5],[0.1,0.2,0.3,0.4,0.5]])
    # test=psdb('test')
    # probes=test.probes
    # print(probes)


    c=[]

    print(2,"m")
