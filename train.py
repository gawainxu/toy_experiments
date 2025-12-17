##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:04:24 2022

@author: zhi
"""

import os
import torch

from datasets.cifar10 import myCIFAR10
from datasets.cifar100 import myCIFAR100
from utils.status import progress_bar
from datasets.utils.utils import mapping
from torch.utils.data import DataLoader


datasets_dict = {"cifar10": myCIFAR10, "cifar100": myCIFAR100}


def training(args, model, method):
    
    sorted_labels, label_mapping = mapping(args.task)
            
    dataset = datasets_dict[args.dataset](root="../../data", classes=sorted_labels,                     
                                          train=True, download=True, args=args)

    data_loader = DataLoader(dataset, batch_size = args.batch_size, 
                             num_workers=args.num_workers, shuffle = True)

    if os.path.isfile(args.init_model_path):
        model.load_state_dict(torch.load(args.init_model_path))

    else:
        print("Train Init Model")
        min_epoch_loss = 1e10
        epoch_loss = 0

        for e in range(args.n_epochs):  
            for i, (inputs, labels) in enumerate(data_loader):
                loss = method.first_task(inputs, labels)
                epoch_loss += loss
                progress_bar(i, data_loader.__len__(), e, 0, loss)
                    
            print("Epoch loss: ", epoch_loss / len(dataset))
            if epoch_loss < min_epoch_loss:
               min_epoch_loss = epoch_loss
               save_path = '../save/' + args.dataset + '_' + args.model + '_' + str(args.task) + '.pth'
               torch.save(model.state_dict(), save_path) 
            epoch_loss = 0
    
    return label_mapping