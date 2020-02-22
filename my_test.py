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
    ##加载训练数据集
    # from roi_data_layer.roidb import combined_roidb
    # from datasets.samplers_ps import sampler
    # from torch.utils.data.sampler import Sampler
    # from roi_data_layer.roibatchLoader import roibatchLoader
    #
    # imdb_name = 'train'
    # batch_size = 2
    # num_workers = 4  # 1
    #
    # imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
    # train_size = len(roidb)
    # print('{:d} roidb entries'.format(len(roidb)))
    #
    # sampler_batch = sampler(train_size, batch_size)
    # dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,imdb.num_classes, training=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                          sampler=sampler_batch, num_workers=num_workers)
    # iters_per_epoch = int(train_size / batch_size)
    # data_iter = iter(dataloader)
    # for step in range(3):
    #     data = next(data_iter)
    #     print('im_data========================',data[0])
    #     print('im_info========================',data[1])
    #     print('gt_boxes========================',data[2])
    #     print('num_boxes========================',data[3])

    # roidb=gt_roidb()
    # for img in roidb:
    #     print('gt[boxes]=============', img['boxes'])
    #     print('gt[im_name]===========', img['im_name'])
    #     print('gt[gt_pids]===========', img['gt_pids'])

    print(2,'d',3)

