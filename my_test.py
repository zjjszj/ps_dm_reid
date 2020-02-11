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
from torch.autograd import Variable
a=torch.tensor([[1,2],[2,2.0]]).cuda()
b=torch.tensor([[1,2],[2,2.0]]).cuda()
c=b.mm(a.t())
print(c)













