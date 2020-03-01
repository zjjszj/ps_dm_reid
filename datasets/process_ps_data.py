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

#错误的方法，使用的是test的数据
def ps_test(model, ps_manager, nums): #nums是图像的个数
    ps_manager.roidb=gt_test_roidb()
    ps_manager.indexs = [i for i in range(len(ps_manager.roidb))]
    model.eval()
    correct = 0
    with torch.no_grad():
        data, target=ps_manager.get_batchData(0,nums)
        data=data.cuda()
        output = model(data).cpu()
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        print('pred=====',pred)
        print('target========',target)
        correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / nums
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, nums, rank1))
    ps_manager.roidb=gt_train_roidb()
    ps_manager.indexs = [i for i in range(len(ps_manager.roidb))]
    return rank1

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

    def get_query_inputs(self):
        q_Image=[]
        test = psdb('test', root_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset')
        probes = test.probes  #[(img_path,box)...]
        for item in probes:
            img_path=item[0]
            box=item[1]
            img=read_pedeImage(img_path)
            pede=img.crop(box)
            q_Image.append(pede)
        q_tensor=TrainTransform()(q_Image)
        q_tensor=torch.stack(q_tensor)

        return q_tensor

    def get_gallery_det_tensor(self, img_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset/Image/SSM'):

        g_det=[]
        g_tensor=[]
        test_roidb=gt_test_roidb()
        for img in test_roidb:
            boxes=img['boxes']
            im_name=img['im_name']
            #boxes = np.hstack((a, np.ones((a.shape[0], 1))))
            g_det.append(boxes)
            image=read_pedeImage(osp.join(img_dir, im_name))
            for box in boxes:
                img_Image = []
                pede_Image=image.crop(box)
                img_Image.append(pede_Image)
                img_Image=TrainTransform()(img_Image)
                g_tensor.append(img_Image)
                del img_Image
                gc.collect()

        return g_det, g_tensor   #[[[tensor],...],...]

    def evaluate(self, model):
        test = psdb('test', root_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset')

        load=False
        save=True
        if load:
            # load
            q_inputs=_load('q_inputs.pkl',r'F:/datasets/reid/')
            g_det=_load('g_det.pkl',r'F:/datasets/reid/')
            g_tensor=_load('g_tensor.pkl',r'F:/datasets/reid/')
            pass
        else:
            #q_inputs=self.get_query_inputs()
            print('begin...get_gallery_det_tensor...')
            g_det, g_tensor=self.get_gallery_det_tensor()
            print('end...get_gallery_det_tensor...')

        if save:
            #save
            #pickle(q_inputs, './evaluate_data', 'g_inputs.pkl')
            print('begin...pickle...')
            pickle(g_det, './evaluate_data', 'g_det.pkl')
            pickle(g_tensor, './evaluate_data', 'g_tensor.pkl')
            print('end...pickle...')
            return




        print('g_tensor size=', sys.getsizeof(g_tensor) / pow(10, 6), 'M')  # M

        #分批次输入到网络，batch_size=64
        q_feat=[]
        batch_size=32
        for i in range(math.ceil(q_inputs.size(0)/batch_size)):
            print('di i batch ',i)
            start=i*batch_size
            end=start+batch_size if (start+batch_size)<q_inputs.size()[0] else q_inputs.size()[0]
            q_feat.extend(model(q_inputs[start:end].cuda()))
            print('q_feat size=',sys.getsizeof(q_feat) / pow(10, 6),'M')  #M
        q_feat=q_feat.numpy()   #[[feat1], ...]


        g_feat=[]
        for img in g_tensor:
            img=torch.stack(img)
            g_feat.append(model(img.cuda()).cpu())
        g_feat=g_feat.numpy()   #[[feat1], ...]

        print('begin run evaluate_search() function......')
        test.evaluate_search(g_det,g_feat,q_feat)


