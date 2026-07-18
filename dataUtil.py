#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:42:16 2021

@author: zhi
"""


classMap = {0: "apples", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
            5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottles",
            10: "bowls", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
            15: "camel", 16: "cans", 17: "castle", 18: "caterpillar", 19: "cattle",
            20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
            25: "couch", 26: "crab", 27: "crocodile", 28: "cups", 29: "dinosaur",
            30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
            35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 
            40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
            45: "lobster", 46: "man", 47: "maple", 48: "motorcycle", 49: "mountain",
            50: "mouse", 51: "mushrooms", 52: "oak", 53: "oranges", 54: "orchids", 
            55: "otter", 56: "palm", 57: "pears", 58: "pickup_truck", 59: "pine",
            60: "plain", 61: "plates", 62: "poppies", 63: "porcupine", 64: "possum",
            65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
            70: "roses", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
            75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
            80: "squirrel", 81: "streetcar", 82: "sunflowers", 83: "pepper", 84: "table", 
            85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
            90: "train", 91: "trout", 92: "tulips", 93: "turtle", 94: "wardrobe",
            95: "whale", 96: "willow", 97: "wolf", 98: "woman", 99: "worm"}

classMap = {v : k for k, v in classMap.items()}

superClasses = [["beaver", "dolphin", "otter", "seal", "whale"],
                ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                ["orchids", "poppies", "roses", "sunflowers", "tulips"],
                ["bottles", "bowls", "cans", "cups", "plates"],
                ["apples", "mushrooms", "oranges", "pears", "peppers"],
                ["clock", "keyboard", "lamp", "telephone", "television"],
                ["bed", "chair", "couch", "table", "wardrobe"],
                ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                ["bear", "leopard", "lion", "tiger", "wolf"],
                ["bridge", "castle", "house", "road", "skyscraper"],
                ["cloud", "forest", "mountain", "plain", "sea"],
                ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                ["fox", "porcupine", "possum", "raccoon", "skunk"],
                ["crab", "lobster", "snail", "spider", "worm"],
                ["baby", "boy", "girl", "man", "woman"],
                ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                ["maple", "oak", "palm", "pine", "willow"],
                ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]]


osr_splits_inliers = {
                  
    "cifar100_marco": [[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
                       [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96],
                       [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97],
                       [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98],
                       [0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46,50,51,55,56,60,61,65,
                        66,70,71,75,76,80,81,85,86,90,91,95,96],
                       [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27, 30, 31, 32,
                        35, 36, 37, 40, 41, 42, 45, 46, 47, 50, 51, 52, 55, 56, 57, 60, 61, 62, 65, 66,
                        67, 70, 71, 72, 75, 76, 77, 80, 81, 82, 85, 86, 87, 90, 91, 92, 95, 96, 97],
                       [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27,
                        28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53,
                        55, 56, 57, 58, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80,
                        81, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 95, 96, 97, 98],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       list(range(100))
                       ],
}


osr_splits_outliers = {

    "cifar100_marco": [
                       #[4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
                       #[3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99]
    ],
}

def pickClass(classIdx):
    
    classNames = superClasses[classIdx]
    classList = []
    for n in classNames:
        classList.append(classMap[n])
        
    return classList


from data_loader import iCIFAR100
from torchvision import transforms

data_root = "../datasets"

num_inlier_classes_mapping = {"cifar100_marco": 20,}


data_function_mapping = {"cifar100_marco": iCIFAR100,}

data_function_mapping_testing = {"cifar100_marco": iCIFAR100}


mean_mapping = {"cifar100_marco": (0.4914, 0.4822, 0.4465),}

std_mapping = {"cifar100_marco": (0.2023, 0.1994, 0.2010),}


image_size_mapping = {"cifar100_marco": 32,}


def label_to_dict(labels, outliers=False):
    label_dict = dict()
    for i, l in enumerate(labels):
        if outliers is False:
            label_dict[str(l)] = i
        else:
            label_dict[str(l)] = 1000

    return label_dict


def get_train_datasets(opt, class_idx=None,):
    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    train_transform = transforms.Compose(
                [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 # transforms.Resize((size, size)),
                 transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),  # !!!!!!
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2),
                 # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),    # !!!!!!!!!!!
                 transforms.ToTensor(),
                 normalize, ])  # normalize,

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]

    if opt.datasets == "svhn":
        train = "train"
    else:
        train = True

    train_dataset = data_fun(root=data_root, train=train,
                             classes=classes, download=True,
                             transform=train_transform, label_dict=label_dict,
                             )
    print("dataset size", len(train_dataset))
    return train_dataset


def get_test_datasets(opt, class_idx = None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    train = False
    test_dataset = data_fun(root=data_root, train=train,
                            classes=classes, download=True, 
                            transform=test_transform, label_dict=label_dict)
    print("dataset size", len(test_dataset))
    return test_dataset


def get_outlier_datasets(opt, class_idx=None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping_testing[opt.datasets]
    label_dict = label_to_dict(osr_splits_outliers[opt.datasets][opt.trail], outliers=True)
    if class_idx is not None:
        classes = [osr_splits_outliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_outliers[opt.datasets][opt.trail]
    print(classes)
    train = False
    outlier_dataset = data_fun(root=data_root, train=train,
                               classes=classes, download=True, 
                               transform=test_transform, label_dict=label_dict)
    print("dataset size", len(outlier_dataset))
    return outlier_dataset