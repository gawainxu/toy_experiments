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

    parser.add_argument('--datasets', type=str, default='cars',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', 'cifar100', "tinyimgnet",
                                 'mnist', "svhn", "cub", "aircraft", "cars", "FUB", "imagenet100"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet50_pretrain", choices=["resnet18", "vgg16", "resnet50_pretrain", "simCNN", "MLP"])
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/cars_models/cars_resnet50_pretrain_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_0,1,2,3_Joint_0.5_0.5_trail_0_128_256_split_128/last.pth")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--trail_outliers", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--start_class", type=int, default=0)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")
    parser.add_argument("--if_merge", type=bool, default=False)

    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument('--temp1', type=float, default=0.05, help='temperature for loss function late')
    parser.add_argument('--temp2', type=float, default=0.01, help='temperature for loss function early')
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--epoch", type=int, default = 600)
    parser.add_argument("--tau_strategy", type=str, default="fixed", choices=["fixed", "fixed_set", "fixed_set_diff", "cosine", "linear", "exp"])
    parser.add_argument("--cosine_period", type=float, default=1.0)
    parser.add_argument("--augmentation_method", type=str, default="vanilia", choices=["vanilia", "upsampling", "mixup"])
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lr", type=str, default=0.01)
    parser.add_argument("--training_bz", type=int, default=600)
    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
    parser.add_argument("--alpha_negative", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_positive", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--intra_inter_mix_positive", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_negative", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--mixup_positive", type=bool, default=False)
    parser.add_argument("--mixup_negative", type=bool, default=False)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--positive_method", type=str, default="no", choices=["min_similarity", "random", "prob_similarity", "no"])
    parser.add_argument("--negative_method", type=str, default="no", choices=["max_similarity", "random", "no"])


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])

    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path = (opt.feature_save + opt.model_name + "_" + str(opt.epoch)
                     + "_" + opt.if_train)

    return opt


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

    model = load_model(opt, model)
    return model


def load_model(opt, model=None):

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        if "fc2" not in k:
            k = k.replace("module.", "")
            new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    return model


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


def normalFeatureReading(data_loader, model, opt):
    
    outputs_backbone = []
    outputs = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if i > opt.break_idx:
            break

        img = img.cuda()
        output, output_encoder = model.fc1(model.encoder(img)), model.encoder(img)

        outputs.append(output.cpu().detach().numpy())
        outputs_backbone.append(output_encoder[-1].cpu().detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, outputs_backbone, labels), f)
        

if __name__ == "__main__":
    
    opt = parse_option()

    model, linear_model = load_model(opt)
    print("Model loaded!!")
    
    featurePaths= []

    train_loader, test_loader = set_loader(opt)

    opt.save_path = opt.save_path_all
    normalFeatureReading(train_loader, model, linear_model, opt)
