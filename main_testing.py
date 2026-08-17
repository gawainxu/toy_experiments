#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:38:04 2024

@author: zhi
"""
from __future__ import print_function

import os
import sys
import argparse
import time
import pickle

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util import AverageMeter
from util import adjust_learning_rate
from util import accuracy
import torch.optim as optim
from dataUtil import osr_splits_inliers, get_train_datasets, get_test_datasets
from networks.resnet_big import SupCEResNet
from networks.vgg import vgg16, vgg11_bn
from networks.LeNet import LeNet5

import matplotlib

matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument("--model_path", type=str, default=None)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet34", "vgg16", "simCNN", "MLP", "lenet"])
    parser.add_argument("--resnet_wide", type=int, default=1, help="factor for expanding channels in wide resnet")
    parser.add_argument('--datasets', type=str, default='cifar100_marco',
                        choices=['cifar10', "tinyimgnet", 'mnist', "svhn", "cifar100_marco"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--trail", type=int, default=0, help="index of repeating training")


    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'

    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    return opt


def set_loader(opt):
    # construct data loader

    train_dataset = get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                              drop_last=True)
    return train_loader, test_loader


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if "resnet" in opt.model:
        model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes)
    elif "vgg" in opt.model:
        model = vgg16(num_classes=opt.num_classes)
    elif "lenet" in opt.model:
        model = LeNet5(num_classes=opt.num_classes)

    model = load_model(opt, model=model)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model


def load_model(opt, model=None):
    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    return model


def val(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)

            # update metric
            acc1, _, _ = accuracy(output, labels)
            top1.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
            idx, len(val_loader), batch_time=batch_time, top1=top1))

    return top1.avg


def main():
    opt = parse_option()

    # build data loader
    _, test_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    acc_val = val(test_loader, model, opt)



if __name__ == '__main__':
    main()
