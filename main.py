"""
TODOs:
    1. feature visualization
    2. metrics
    3. optimal training configs
    4. cnn feature localization on original images
"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import argparse

from datasets.cifar10 import myCIFAR10
from datasets.mnist import myMNIST
from methods.plain import plain
from models.vgg import VGG
import models.vgg as vgg
from models.resnet_big import SupCEResNet
from utils.args import add_experiment_args, add_management_args, add_rehearsal_args
from utils.args import add_cifar100_args, add_cifar10_args, add_mnist_args
from utils.args import add_plain_args, add_resnet_args, add_vgg_args
from train import training
from testing import testing
from features.feature_reading import normalFeatureReading

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


models_args_dict = {"vgg": add_vgg_args, "resnet": add_resnet_args}

datasets_args_dict = {"cifar10": add_cifar10_args, "cifar100": add_cifar100_args,
                      "mnist": add_mnist_args}



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('argument for training')
    add_experiment_args(parser)
    add_management_args(parser)
    add_plain_args(parser)
    opts = parser.parse_args()
    
        
    if opts.model in models_args_dict.keys():
        print("Method parameters added")
        models_args_dict[opts.model](parser)
        opts = parser.parse_args()
        
    if opts.dataset in datasets_args_dict.keys():
        print("Dataset parameters added")
        datasets_args_dict[opts.dataset](parser)
        opts = parser.parse_args()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.task = eval(opts.task)
    
    if opts.if_train == "training":
        
        if opts.model == "vgg":
           model = VGG(vgg.vgg_cfgs[opts.vgg_type], num_classes=len(opts.task), args=opts)
        elif opts.model == "resnet":
           model = SupCEResNet(num_classes=len(opts.task))                     
       
        method = plain(opts, model, device)

        label_mapping = training(opts, model, method)
        

    elif opts.if_train == "testing":

        if opts.model == "vgg":
           model = VGG(vgg.vgg_cfgs[opts.vgg_type], num_classes=len(opts.task), args=opts)
        elif opts.model == "resnet":
           model = SupCEResNet(num_classes=len(opts.task))

        method = plain(opts, model, device)
        acc = testing(model, opts, method)
        print("acc ", acc)
        

    elif opts.if_train == "feature":

        if opts.model == "vgg":
           model = VGG(vgg.vgg_cfgs[opts.vgg_type], num_classes=len(opts.task), args=opts)
        elif opts.model == "resnet":
           model = SupCEResNet(num_classes=len(opts.task))

        print(model)
        normalFeatureReading(model, opts)
        
    else:
        raise Exception('I do not know what to do!')
        