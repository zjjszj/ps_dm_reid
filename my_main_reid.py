# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
#from models.networks import ResNetBuilder, IDE, Resnet, BFE
from models.networks_my import ResNetBuilder, IDE, Resnet, ResNet_openReid,BFE_Finally
from trainers.evaluator import ResNetEvaluator
from trainers.trainer import cls_tripletTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin, OIMLoss
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
#加载ps数据集
import random
from datasets.process_ps_data import ps_data_manager


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')


    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    ##加载训练数据集
    ps_manager=ps_data_manager('train_pedes')

    print('initializing model ...')
    model = BFE_Finally(5532, 1.0, 0.33)  # dataset.num_train_pids

    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        #state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if opt.loss == 'triplet':
        embedding_criterion = TripletLoss(opt.margin)
    elif opt.loss == 'lifted':
        embedding_criterion = LiftedStructureLoss(hard_mining=True)
    elif opt.loss == 'weight':
        embedding_criterion = Margin()
    #oim
    # elif opt.loss=='oim':
    #     embedding_criterion_global = OIMLoss(num_features=512, num_classes=751)
    #     embedding_criterion_drop = OIMLoss(num_features=1024, num_classes=751)
    elif opt.loss=='oim+triplet':
        oim_criterion=OIMLoss(num_features=128,num_classes=5532)
        triplet_criterion = TripletLoss(opt.margin)



    #原始的+oim
    # def criterion(triplet_y, softmax_y, labels):   #输出向量[全局，局部]、输出得分、标签
    #     if opt.loss=='oim':
    #         loss= [embedding_criterion_global(triplet_y[0], labels)[0]]+\
    #                  [embedding_criterion_drop(triplet_y[1], labels)[0]]
    #         loss=loss[0]
    #         #print('loss==========',loss) 6.6214
    #     else:
    #         losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
    #                      [xent_criterion(output, labels) for output in softmax_y]
    #         loss = sum(losses)
    #     return loss




    ##adding global and local vector.Using oim
    elif opt.loss=='oim':
        oim_criterion = OIMLoss(num_features=128, num_classes=5532)  #num_classes=751 market1501

    def criterion(triplet_y, labels):  # 输出向量[全局，局部]、输出得分、标签
        if opt.loss=='oim':
            #losses = embedding_criterion(triplet_y, labels)[0]  #单分支
            loss = [oim_criterion(output1, labels)[0] for output1 in triplet_y]
            losses = sum(loss)/2 #损失值太大了
        elif opt.loss=='oim+triplet':    #triplet损失函数对训练数据有要求
            loss = [oim_criterion(output1, labels)[0] for output1 in triplet_y] + \
                [triplet_criterion(output2, labels)[0] for output2 in triplet_y]
            losses = sum(loss)/4   #损失值太大了
        return losses
    ##end

    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)


    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep):
        for p in optimizer.param_groups:
            lr=p['lr']
        if ep < 5:
            lr = lr-0.00009
        elif ep < 200:
            lr = 1e-4
        elif ep < 300:
            lr = 1e-4
        else:
            lr = 1e-4
        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0

    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            adjust_lr(optimizer, epoch + 1)

        reid_trainer.train(epoch, ps_manager=ps_manager)

        # # skip if not save model
        # if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
        #     if opt.mode == 'class':
        #         rank1 = test(model, queryloader)
        #     else:
        #         rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
        #     is_best = rank1 > best_rank1
        #     if is_best:
        #         best_rank1 = rank1
        #         best_epoch = epoch + 1
        #
        #     if use_gpu:
        #         state_dict = model.module.state_dict()
        #     else:
        #         state_dict = model.state_dict()
        #     save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
        #         is_best=is_best, save_dir=opt.save_dir,
        #         filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

        #skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            rank1 = ps_manager.evaluate(model)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                pass
            #     state_dict = model.state_dict()
            # save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
            #     is_best=is_best, save_dir=opt.save_dir,
            #     filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))

def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1

if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()
