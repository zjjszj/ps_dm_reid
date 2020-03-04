from os import path as osp
import _pickle as cPickle
from PIL import Image
import os.path as osp
from torchvision import transforms as T
import random
import torch
from datasets.psdb import psdb
import math
import numpy as np
import sys
import errno
import os
import gc

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

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def pickle(data, save_dir, file_name):
    mkdir_if_missing(save_dir)
    file_path=osp.join(save_dir, file_name)
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, 0)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def _load(fname, output_dir):
    fpath = osp.join(output_dir, fname)
    assert osp.isfile(fpath), "Must have extracted detections and " \
                              "features first before evaluation"
    return unpickle(fpath)

def gt_train_roidb():
    #cache_file = 'E:/data/cache/psdb_train_gt_roidb.pkl'  #项目的根目录  用于pycharm
    cache_file = '/kaggle/input/psdb-train-roidb/psdb_train_gt_roidb.pkl'  #项目的根目录   用于kaggle
    if osp.isfile(cache_file):
        roidb = unpickle(cache_file)
        return roidb

def gt_test_roidb():
    #cache_file = 'E:/data/cache/psdb_test_gt_roidb.pkl'  #项目的根目录  用于pycharm
    cache_file = '/kaggle/input/psdb-test-roidb/psdb_test_gt_roidb.pkl'  #项目的根目录   用于kaggle
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
    def __call__(self, x):  #x:[pede,...]
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
        self.roidb = gt_train_roidb()
        #self.pids2label = self.pids_to_label()
        self.indexs = [i for i in range(len(self.roidb))]

    def pids_to_label(self):
        # 制作label
        pids_container = set()
        for i in range(len(self.roidb)):
            img = self.roidb[i]
            img_gt_pids = img['gt_pids']
            for j in range(len(img_gt_pids)):
                if img_gt_pids[j] != -1:
                    pids_container.add(img_gt_pids[j])
        pid2label = {pid: label for label, pid in enumerate(pids_container)}
        return pid2label

    def get_batchData(self, i_batch, batch_size):
        """
        get batch_size images per time. convert those images to pedes for model inputs.
        :param i_batch: index of batch
        :param batch_size:
        :return:
        """
        pedes_batch_x = []
        pedes_batch_y = []
        indexs_batch = [self.indexs[i] for i in range(i_batch * batch_size,
            i_batch * batch_size + batch_size if i_batch * batch_size <= len(self.roidb) else len(self.roidb))]
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
        """

        :param im_name: image name
        :param boxes: all boxes per img
        :param gt_pids: all pids per img
        :param img_dir:
        :return: [Image type pede...], [pede label...]
        """
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

    def get_query_feat(self, model):
        q_feat=[]
        test = psdb('test', root_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset')
        probes = test.probes  #[(img_path,box)...]
        batch_size=16
        with torch.no_grad():
            for i in range(math.ceil(len(probes)/batch_size)):
                start=i*batch_size
                end=start+batch_size if (start+batch_size)<len(probes) else len(probes)
                batch_probes=probes[start:end]
                pedes_Image=[]
                for probe in batch_probes:
                    img_path=probe[0]
                    box=probe[1]
                    img=read_pedeImage(img_path)
                    pede=img.crop(box)
                    pedes_Image.append(pede)
                pedes_list=TrainTransform()(pedes_Image)  #TrainTransform返回为一个tensor的list
                q_feat.extend(np.asarray(model(torch.stack(pedes_list).cuda()).cpu()))
                #del pedes_Image, pedes_list, batch_probes
                #gc.collect()
        return q_feat  #q_feat: list [[array],...]

    def get_gallery_det(self):
        g_det=[]
        test_roidb=gt_test_roidb()
        for img in test_roidb:
            boxes=img['boxes']
            g_det.append(boxes)
        return g_det

    def get_gallery_feat(self, model, img_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset/Image/SSM'):

        test_roidb=gt_test_roidb()
        g_feat=[]
        with torch.no_grad():     #没有该句，显存不够！！1.5G-> >15G
            for img in test_roidb:
                img_Image = []
                boxes=img['boxes']
                im_name=img['im_name']
                image=read_pedeImage(osp.join(img_dir, im_name))
                for box in boxes:
                    pede_Image=image.crop(box)
                    img_Image.append(pede_Image)
                img_list = TrainTransform()(img_Image)
                g_feat.append(np.asarray(model(torch.stack(img_list).cuda()).cpu()))
                del img_Image, img_list
                gc.collect()
        return g_feat   #[[[array],...],...]

    def evaluate(self, model):
        test = psdb('test', root_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset')
        model.eval()
        print('begin...get_gallery_det...')
        g_det=self.get_gallery_det()
        print()
        print('begin...get_query_feat...')
        q_feat=self.get_query_feat(model)
        print()
        print('begin...get_gallery_feat...')
        g_feat=self.get_gallery_feat(model)
        print()
        print('begin run evaluate_search() function......')
        return test.evaluate_search(g_det,g_feat,q_feat)

    # 使用一批训练的数据进行测试
    def ps_test(self, model, test_data):
        model.eval()
        correct = 0
        with torch.no_grad():
            data, target=test_data
            output = model(data.cuda()).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        rank1 = 100. * correct / len(test_data.data)
        print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_data.data), rank1))
        return rank1