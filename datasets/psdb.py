import json
import os
import os.path as osp

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve

#import datasets
#from datasets.imdb import imdb
# from fast_rcnn.config import cfg
#from utils import cython_bbox
from datasets.imdb import imdb

#utils/__init__.py
import _pickle as cPickle
import re


def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, 0)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


#输出信息保存到文件中
# import sys
# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
# sys.stdout = Logger('a.txt')


class psdb(imdb):
    # r'F:datasets/reid/CUHK-SYSU_nomacosx/dataset'
    def __init__(self, image_set, root_dir=r'/kaggle/input/cuhk-sysu/CUHK-SYSU_nomacosx/dataset'):
        super(psdb, self).__init__('psdb_' + image_set)
        self._image_set = image_set
        # self._root_dir = self._get_default_path() if root_dir is None \
        #                  else root_dir
        self._root_dir = root_dir
        self._data_path = osp.join(self._root_dir, 'Image', 'SSM')
        self._classes = ('__background__', 'person')
        self._image_index = self._load_image_set_index()
        self._probes = self._load_probes()
        self._roidb_handler = self.gt_roidb
        assert osp.isdir(self._root_dir), \
                "PSDB does not exist: {}".format(self._root_dir)
        assert osp.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = osp.join(self._data_path, index)
        assert osp.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')  #cache_path:E:\AI\data\cache
        if osp.isfile(cache_file):
            roidb = unpickle(cache_file)
            return roidb

        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, __, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, \
                'Warning: {} has no valid boxes.'.format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print('Warning: person {} box {} cannot find in Images'.format(pid, box))

        # Load all the train / test persons and label their pids from 0 to N-1
        # Assign pid = -1 for unlabeled background people

        if self._image_set == 'train':
            print('train=============================================')
            train = loadmat(osp.join(self._root_dir,
                                     'annotation/test/train_test/Train.mat'))
            train = train['Train'].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        else:
            print('test============================================')
            test = loadmat(osp.join(self._root_dir,
                                    'annotation/test/train_test/TestG50.mat'))
            test = test['TestG50'].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item['Query'][0,0][0][0])
                box = item['Query'][0,0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
                # gallery
                gallery = item['Gallery'].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0: break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)

        # Construct the gt_roidb
        gt_roidb = []
        #为self.image_index排序，生成按图像名升序的roidb
        # def sort_key(s):
        #     # sort_strings_with_embedded_numbers
        #     re_digits = re.compile(r'(\d+)')
        #     pieces = re_digits.split(s)  # 切成数字与非数字
        #     pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
        #     return pieces
        # self.image_index.sort(key=sort_key)
        for im_name in self.image_index:         #im_name:图像名
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[im_name]
            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)
            gt_roidb.append({
                'boxes': boxes,
                #'gt_classes': gt_classes,
                #'gt_overlaps': overlaps,
                'gt_pids': pids,
                'im_name':im_name
                #'flipped': False
            })
        pickle(gt_roidb, cache_file)
        print("wrote gt roidb to {}".format(cache_file))
        print('function gt_roidb()->len(gt_roidb)={0}'.format(len(gt_roidb)))
        return gt_roidb

    def evaluate_detections(self, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                            labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image

        det_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.num_images == len(gallery_det)
        gt_roidb = self.gt_roidb()
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(gt_roidb, gallery_det):
            gt_boxes = gt['boxes']
            if labeled_only:
                inds = np.where(gt['gt_pids'].ravel() > 0)[0]
                if len(inds) == 0: continue
                gt_boxes = gt_boxes[inds]
            det = np.asarray(det)
            print('det.shape={0}'.format(det.shape))  #det.shape=(72, 5)
            print('det={0}'.format(det))
            print('det[:, 4].ravel()={0}'.format(det[:, 4].ravel()))
            print('det[:, 4].ravel() >= 0.5={0}'.format(det[:, 4].ravel() >= 0.5))
            inds = np.where(det[:, 4].ravel() >= 0.5)[0]  #det_thresh
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thresh)
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate

        print ('{} detection:'.format('labeled only' if labeled_only else  'all'))
        print('  recall = {:.2%}'.format(det_rate))
        if not labeled_only: print( '  ap = {:.2%}'.format(ap))

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                        det_thresh=0.5, gallery_size=50, dump_json=None):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        det_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                            -1 for using full set
        dump_json (str): Path to save the results as a JSON file or None
        """
        assert self.num_images == len(gallery_det)
        assert self.num_images == len(gallery_feat)
        assert len(self.probes) == len(probe_feat)



        # TODO: support evaluation on training split
        use_full_set = gallery_size == -1
        fname = 'TestG{}'.format(gallery_size if not use_full_set else 50)
        protoc = loadmat(osp.join(self._root_dir, 'annotation/test/train_test',
                                  fname + '.mat'))[fname].squeeze()

        # mapping from gallery image to (det, feat)
        name_to_det_feat = {}
        for name, det, feat in zip(self._image_index,
                                   gallery_det, gallery_feat):
            #使用gt
            # scores = det[:, 4].ravel()
            # inds = np.where(scores >= det_thresh)[0]
            inds=[i for i in range(len(det))]

            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': self._data_path, 'results': []}
        for i in range(len(self.probes)):              #1
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()  #npType.ravel() 将多维数据转为1维
            # Ignore the probe image
            probe_imname = str(protoc['Query'][i]['imname'][0,0][0])
            probe_roi = protoc['Query'][i]['idlocate'][0,0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for item in protoc['Gallery'][i].squeeze():
                gallery_imname = str(item[0][0])
                # some contain the probe (gt not empty), some not
                gt = item[1][0].astype(np.int32)
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat: continue
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({'img': str(gallery_imname),
                                     'roi': map(float, list(gt))})
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                for gallery_imname in self._image_index:
                    if gallery_imname in tested: continue
                    if gallery_imname not in name_to_det_feat: continue
                    det, feat_g = name_to_det_feat[gallery_imname]
                    # get L2-normalized feature matrix NxD
                    assert feat_g.size == np.prod(feat_g.shape[:2])
                    feat_g = feat_g.reshape(feat_g.shape[:2])
                    # compute cosine similarities
                    sim = feat_g.dot(feat_p).ravel()
                    # guaranteed no target probe in these gallery images
                    label = np.zeros(len(sim), dtype=np.int32)
                    y_true.extend(list(label))
                    y_score.extend(list(sim))
                    imgs.extend([gallery_imname] * len(sim))
                    rois.extend(list(det))
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                 average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # 4. Save result for JSON dump
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': map(float, list(probe_roi)),
                         'probe_gt': probe_gt,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': map(float, list(rois[inds[k]])),
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                })
            ret['results'].append(new_entry)

        print ('search ranking:')
        print ('  mAP = {:.2%}'.format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print ('  top-{:2d} = {:.2%}'.format(k, accs[i]))

        if dump_json is not None:
            if not osp.isdir(osp.dirname(dump_json)):
                os.makedirs(osp.dirname(dump_json))
            with open(dump_json, 'w') as f:
                json.dump(ret, f)
        return accs[0]

    def evaluate_cls(self, detections, pid_ranks, pid_labels,
                     det_thresh=0.5):
        """
        detections (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        pid_ranks (list of ndarray): n_det x top_k cls scores per image
        pid_labels (list of ndarray): n_det x 1 ground truth identities

        det_thresh (float): filter out gallery detections whose scores below this
        """
        assert len(detections) == len(pid_ranks)
        assert len(detections) == len(pid_labels)

        # Get the num of identities in the imdb
        gt_roidb = self.gt_roidb()
        max_pid = 0
        for item in gt_roidb:
            max_pid = max(max_pid, max(item['gt_pids']))

        # In the extracted pid_labels:
        #   -1 for unlabeled person,
        #   {0, 1, ..., max_pid-1} for labeled person
        #   max_pid for background clutter
        count_ul, count_lb, count_bg = 0, 0, 0
        y_pred, y_true = [], []
        for dets, ranks, labels in zip(detections, pid_ranks, pid_labels):
            assert len(dets) == len(ranks)
            assert len(dets) == len(labels)
            for det, rank, label in zip(dets, ranks, labels):
                if det[-1] < det_thresh: continue
                label = int(round(label))
                if label == -1:
                    count_ul += 1
                    continue
                elif label == max_pid:
                    count_bg += 1
                    continue
                else:
                    count_lb += 1
                    y_pred.append(rank)
                    y_true.append(label)

        # some statistics
        print ('classifiction:')
        print ('  number of background clutter =', count_bg)
        print ('  number of unlabeled =', count_ul)
        print ('  number of labeled =', count_lb)

        # top-k classification accuracies
        correct = np.asarray(y_pred) == np.asarray(y_true)[:, np.newaxis]
        for top_k in [1, 5, 10]:
            acc = correct[:, :top_k].sum(axis=1).mean()
            print ('  top-{} accuracy = {:.2%}'.format(top_k, acc))


    # def _get_default_path(self):
    #     return osp.join(cfg.DATA_DIR, 'psdb', 'dataset')

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self._root_dir, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]
        if self._image_set == 'test': return test
        # all images
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        return list(set(all_imgs) - set(test))

    def _load_probes(self):
        """
        Load the list of (img, roi) for probes. For test split, it's defined
        by the protocol. For training split, will randomly choose some samples
        from the gallery as probes.
        """
        protoc = loadmat(osp.join(self._root_dir,
            'annotation/test/train_test/TestG50.mat'))['TestG50'].squeeze()
        probes = []
        for item in protoc['Query']:
            #im_name = osp.join(self._data_path, str(item['imname'][0,0][0]))
            im_name = str(item['imname'][0,0][0])
            roi = item['idlocate'][0,0][0].astype(np.int32)
            roi[2:] += roi[:2]
            probes.append((im_name, roi))
        return probes

    def pede_train_data(self):
        """
        generate train data: {{'pid':pid, 'im_name':im_name, 'boxes':boxes},...}
        :return: train data
        """
        cache_file = osp.join(self.cache_path, self.name + '_pedes.pkl')
        if osp.isfile(cache_file):
            psdb_train_pedes = unpickle(cache_file)
            return psdb_train_pedes
        train_roidb = self.roidb
        pede_train_data=[]
        for i in range(5532):  #5532个行人id
            pid_imnames = []
            pid_boxes = []
            for img in train_roidb:
                im_name=img['im_name']
                boxes=img['boxes']
                pids=img['gt_pids']
                for j in range(len(pids)):
                    if pids[j]==i:
                        pid_imnames.append(im_name)
                        pid_boxes.append(boxes[j])
                        break
            pid_boxes=np.asarray(pid_boxes)
            pid_boxes=pid_boxes.reshape(pid_boxes.shape[0], 4)
            pede_train_data.append({
                'pid': i,
                'im_name': pid_imnames,
                'boxes': pid_boxes
            })
        pickle(pede_train_data, cache_file)   #E:\AI\data\cache
        return pede_train_data

    def pids_to_label(self):
        # 制作label
        pids_container = set()
        for i in range(len(self.roidb)):
            img = self.roidb[i]
            img_gt_pids = img['gt_pids']
            for j in range(len(img_gt_pids)):
                if img_gt_pids[j] != -1:
                    pids_container.add(img_gt_pids[j])
        for i in pids_container:
            print(i)





if __name__ == '__main__':
    #from datasets.psdb import psdb

    roidb=psdb('test')
    rois=roidb.gt_roidb()
    for img in rois:
        print(img['im_name'])
        # if (img['im_name'] == 's11202.jpg'):
        #     print(img['im_name'])
    # print(len(roidb))
    # d=psdb('test',root_dir=r'F:\datasets\reid\CUHK-SYSU_nomacosx\dataset')
    # test=d.gt_roidb()
    #print(len(roidb))



    #from IPython import embed; embed()       #调式时使用

    #求query和gallery
    # test = loadmat(osp.join(r'F:datasets/reid/CUHK-SYSU_nomacosx/dataset','annotation/test/train_test/TestG50.mat'))
    # test = test['TestG50'].squeeze()
    # for i in range(2900):
    #     # Ignore the probe image
    #     probe_imname = str(test['Query'][i]['imname'][0,0][0])
    #     print('probe_imname==',probe_imname)
    #     # probe_roi = protoc['Query'][i]['idlocate'][0,0][0].astype(np.int32)
    #     # probe_roi[2:] += probe_roi[:2]
    #     # 1. Go through the gallery samples defined by the protocol
    #     for item in test['Gallery'][i].squeeze():
    #         gallery_imname = str(item[0][0])
    #         print('gallery_imname===',gallery_imname)
    #test evaluate_search()
    # gallery_det=np.array([[1,2,3,4,5],[1,2,3,4,5]])
    # gallery_feat=np.array([[0.1,0.2,0.3,0.4,0.5],[0.1,0.2,0.3,0.4,0.5]])
    # query_feat=np.array([[0.11,0.22,0.33,0.44]])
    # d = psdb('test')
    # d.evaluate_search(gallery_det,gallery_feat,query_feat)

    #test _load_probes()

    #test 'Person.mat'


