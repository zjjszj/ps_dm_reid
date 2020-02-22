from os import path as osp
import _pickle as cPickle
import os
import re
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms as T
import random
from torch.utils.data import DataLoader
import torch

class Cutout(object):
    def __init__(self, probability=0.5, size=64, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.size = size

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        h = self.size
        w = self.size
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def gt_roidb():
    #cache_file = 'E:/data/cache/psdb_train_gt_roidb.pkl'  #项目的根目录  用于pycharm
    cache_file = '/kaggle/input/psdb-train-roidb/psdb_train_gt_roidb.pkl'  #项目的根目录   用于kaggle
    if osp.isfile(cache_file):
        roidb = unpickle(cache_file)
        return roidb


def read_pedeImage(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def get_dataset():
    roidb=gt_roidb()
    return roidb





# class PS_Data(Dataset):
#     def __init__(self, dataset, transform):
#         self.dataset = dataset
#         self.transform = transform
#
#     def __getitem__(self, item):
#         im_name = self.dataset[item]['im_name']
#         boxes = self.dataset[item]['boxes']
#         gt_pids = self.dataset[item]['gt_pids']
#         pedes_x, pedes_y = img_process(im_name,boxes,gt_pids)
#         if self.transform is not None:
#             pedes_x = self.transform(pedes_x)
#         return pedes_x, pedes_y
#
#     def __len__(self):
#         return len(self.dataset)


class TrainTransform:
    def __call__(self, x):
        ret=[]
        for pede in x:
            pede = T.Resize((128, 64))(pede)
            pede = T.RandomHorizontalFlip()(pede)
            pede = T.ToTensor()(pede)
            pede = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(pede)
            #Cutout预处理
            pede = Cutout(probability = 0.5, size=64, mean=[0.0, 0.0, 0.0])(pede)   #[3, 128, 64]
            ret.append(pede)
        return ret


class ps_data_manager:

    def set_attr(self):
        self.roidb = gt_roidb()
        #self.pids2label = self.pids_to_label()
        self.indexs = [i for i in range(len(self.roidb))]

    def pids_to_label(self):
        # 制作label
        pids_container = set()
        for i in range(len(self.roidb)):
            img = self.roidb[i]
            img_gt_pids = img['gt_pids']
            print('im_name===',img['im_name'])
            for j in range(len(img_gt_pids)):
                if img_gt_pids[j] != -1:
                    print('img_gt_pids[j]===',img_gt_pids[j])
                    pids_container.add(img_gt_pids[j])
        pid2label = {pid: label for label, pid in enumerate(pids_container)}
        return pid2label

    def get_batchData(self, i_batch, batch_size):
        pedes_batch_x = []
        pedes_batch_y = []
        indexs_batch = [self.indexs[i] for i in range(i_batch * batch_size,
                                                 i_batch * batch_size + batch_size if i_batch * batch_size <= len(
                                                     self.roidb) else len(self.roidb))]
        # print(indexs_batch)
        for item in indexs_batch:
            im_name = self.roidb[item]['im_name']
            # print(im_name)
            boxes = self.roidb[item]['boxes']
            gt_pids = self.roidb[item]['gt_pids']
            pedes_x_Image, pedes_y = self.img_process(im_name, boxes, gt_pids)
            pedes_x = TrainTransform()(pedes_x_Image)
            pedes_batch_x.extend(pedes_x)
            pedes_batch_y.extend(pedes_y)
        pedes_batch_x = torch.stack(pedes_batch_x)
        pedes_batch_y = torch.tensor(pedes_batch_y, dtype=torch.long)
        # print(pedes_batch_x.size())
        # print(pedes_batch_y)
        # print(pedes_batch_x)
        return pedes_batch_x, pedes_batch_y

    def img_process(self, im_name, boxes, gt_pids, img_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset/Image/SSM'):
        pedes_x = []
        pedes_y = []
        image = read_pedeImage(osp.join(img_dir, im_name))
        for i in range(len(gt_pids)):
            if gt_pids[i] != -1:
                boxe = boxes[i]
                pede = image.crop(boxe)
                pedes_x.append(pede)
                pedes_y.append(gt_pids[i])  #self.pids2label[gt_pids[i]]
        return pedes_x, pedes_y