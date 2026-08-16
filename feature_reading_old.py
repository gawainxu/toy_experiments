#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import platform
import sys
BASE_PATH = "/home/sysgen/Jiawen/causal_OSR"
sys.path.append(BASE_PATH) 

import argparse
import torch
import pickle

from networks.resnet_big import SupCEResNet
from  networks.vgg import vgg16
from networks.LeNet import LeNet5

from dataUtil import osr_splits_inliers, get_train_datasets, get_test_datasets


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar100_marco',
                        choices=['cifar100_marco'], help='dataset')
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "vgg16", "resnet50_pretrain"])
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--model_trail", type=int, default=0)
    parser.add_argument("--trail", type=int, default=0, help="data trail")
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument("--feature_save", type=str, default="/features/")

    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    opt = parser.parse_args()

    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.model_trail])

    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    opt.save_path = opt.feature_save + opt.model_name + "_data_" + str(opt.trail) + "_" + opt.if_train

    return opt


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes)
    model = load_model(opt, model=model)

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


def set_loader(opt):

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


def normalFeatureReading(data_loader, model, opt):
    
    outputs_backbone = []
    outputs = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)

        img = img.cuda()
        output, output_encoder = model.fc1(model.encoder(img)), model.encoder(img)

        outputs.append(output.cpu().detach().numpy())
        outputs_backbone.append(output_encoder[-1].cpu().detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, outputs_backbone, labels), f)
        

if __name__ == "__main__":
    
    opt = parse_option()

    model = set_model(opt)
    print("Model loaded!!")

    train_loader, test_loader = set_loader(opt)

    opt.save_path = opt.save_path_all
    normalFeatureReading(train_loader, model, linear_model, opt)
