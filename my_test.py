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

from torch import autograd
from torch.autograd import Variable

class Exp(autograd.Function):
    @staticmethod
    def forward(ctx, i,m):
        result = i.exp()
        ctx.save_for_backward(result,m)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        grad_output=2
        result, m= ctx.saved_tensors
        m=m*2
        print('m=====',m)
        return grad_output * result,None


a=torch.tensor(2.0,requires_grad=True)
v=Exp.apply(a,torch.tensor(2))
v.backward()
print(a.grad)







