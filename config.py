# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'Market1501'
    datatype = 'person'
    mode = 'retrieval'
    # optimization options
    loss = 'oim'  #triplet  oim  oim+triplet
    num_instances=3  #when loss=oim+triplet
    oim_scalar=30
    optim = 'adam'  #adam  sgd
    max_epoch =1    #400
    train_batch = 16
    nums_pedes=16
    test_batch = 32
    adjust_lr = False
    lr = 0.0001
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    random_crop = False
    margin = None
    num_instances = 4
    num_gpu = 1
    evaluate = False
    savefig = None 
    re_ranking = False

    # model options
    model_name = 'bfe'  # triplet, softmax_triplet, bfe, ide
    last_stride = 1
    pretrained_model = None
    
    # miscs
    print_freq = 30
    #评估并保存的伦数
    eval_step = 2
    save_dir = './pytorch-ckpt/market'
    workers = 4
    start_epoch = 0
    best_rank = -np.inf

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            if 'cls' in self.dataset:
                self.mode='class'
            if 'market' in self.dataset or 'cuhk' in self.dataset or 'duke' in self.dataset:
                self.datatype = 'person'
            elif 'cub' in self.dataset:
                self.datatype = 'cub'
            elif 'car' in self.dataset:
                self.datatype = 'car'
            elif 'clothes' in self.dataset:
                self.datatype = 'clothes'
            elif 'product' in self.dataset:
                self.datatype = 'product'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()
