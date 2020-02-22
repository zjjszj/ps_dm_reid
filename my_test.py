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


from torch.utils.data import DataLoader
if __name__ == '__main__':

    a=set()
    a.add('one')
    a.add('two')
    pid2label = {pid: label for label, pid in enumerate(a)}
    for i,j in enumerate(pid2label):
        print('i=',i,'j=',j)