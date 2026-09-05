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

#classMap = {v : k for k, v in classMap.items()}

superClasses = {"beaver" : 0, "dolphin": 0, "otter" : 0, "seal": 0, "whale": 0,
                "aquarium_fish": 1, "flatfish": 1, "ray": 1, "shark": 1, "trout": 1,
                "orchids": 2, "poppies": 2, "roses": 2, "sunflowers": 2, "tulips": 2,
                "bottles": 3, "bowls": 3, "cans": 3, "cups": 3, "plates": 3,
                "apples": 4, "mushrooms" : 4, "oranges" : 4, "pears" : 4, "peppers" : 4,
                "clock": 5, "keyboard": 5, "lamp": 5, "telephone": 5, "television": 5,
                "bed": 6, "chair": 6, "couch": 6, "table": 6, "wardrobe": 6,
                "bee": 7, "beetle": 7, "butterfly": 7, "caterpillar": 7, "cockroach": 7,
                "bear": 8, "leopard": 8, "lion": 8, "tiger": 8, "wolf": 8,
                "bridge": 9, "castle": 9, "house": 9, "road": 9, "skyscraper": 9,
                "cloud": 10, "forest": 10, "mountain": 10, "plain": 10, "sea": 10,
                "camel": 11, "cattle": 11, "chimpanzee": 11, "elephant": 11, "kangaroo": 11,
                "fox": 12, "porcupine": 12, "possum": 12, "raccoon": 12, "skunk": 12,
                "crab": 13, "lobster": 13, "snail": 13, "spider": 13, "worm": 13,
                "baby": 14, "boy": 14, "girl": 14, "man": 14, "woman": 14,
                "crocodile": 15, "dinosaur": 15, "lizard": 15, "snake": 15, "turtle": 15,
                "hamster": 16, "mouse": 16, "rabbit": 16, "shrew": 16, "squirrel": 16,
                "maple": 17, "oak": 17, "palm": 17, "pine": 17, "willow": 17,
                "bicycle": 18, "bus": 18, "motorcycle": 18, "pickup_truck": 18, "train": 18,
                "lawn_mower": 19, "rocket": 19, "streetcar": 19, "tank": 19, "tractor": 19}


osr_splits_inliers = {
    "cifar100_marco": [[4, 1, 54, 9, 0, 22, 5, 6, 3, 12], # 0
                       [30, 32, 62, 10, 51, 39, 20, 7, 42, 17], # 1
                   [55, 67, 70, 16, 53, 40, 25, 14, 43, 37], # 2
                   [72, 73, 82, 28, 57, 86, 84, 18, 88, 68], # 3
                   [95, 91, 92, 61, 83, 87, 94, 24, 97, 76], # 4

                   [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 17, 20, 22, 30, 32, 39, 42, 51, 54, 62],   # 5, 0+1
                   [0, 1, 3, 4, 5, 6, 9, 12, 14, 16, 22, 25, 37, 40, 43, 53, 54, 55, 67, 70],    # 6, 0+2
                   [7, 10, 14, 16, 17, 20, 25, 30, 32, 37, 39, 40, 42, 43, 51, 53, 55, 62, 67, 70], # 7, 1+2

                   [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16, 17, 20, 22, 25, 30, 32, 37, 39, 40, 42, 43, 51, 53, 54, 55, 62, 67, 70], # 8, 0+1+2

                   [23, 15, 34, 26, 2, 27, 36, 47, 8, 41],   # 9, far semantic 0
                   [33, 19, 63, 45, 11, 29, 50, 52, 13, 69], # 10, far semantic 1

                   [23, 15, 34, 26, 2, 86, 84, 18, 88, 68],   # 11, moderate semantic 0
                   [33, 19, 63, 45, 11, 87, 94, 24, 97, 76], # 12, moderate semantic 1

                   [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 22, 23, 26, 27, 34, 36, 41, 47, 54], # 13, random init 0
                   [7, 10, 11, 13, 17, 19, 20, 29, 30, 32, 33, 39, 42, 45, 50, 51, 52, 62, 63, 69], # 14, random init 1
                   [14, 16, 21, 25, 35, 37, 40, 43, 44, 48, 49, 53, 55, 56, 64, 65, 67, 70, 77, 81], # 15, random init 2

                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    72, 73, 74, 77, 78, 79, 81, 82, 84, 85, 86, 88], # 16 G0 + G1 + G2 + G3

                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23,
                   25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 62, 63, 64, 65, 67, 69, 70, 77, 81],  # 17, G0 + G1 + G2

                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 22, 23, 26, 27, 29,
                        30, 32, 33, 34, 36, 39, 41, 42, 45, 47, 50, 51, 52, 54, 62, 63, 69],  # 18, G0 + G1

                   [72, 73, 82, 28, 57, 86, 84, 18, 88, 68, 60, 31, 66, 79, 46, 78, 74, 59, 58, 85],  # 19, G3
                   [55, 67, 70, 16, 53, 40, 25, 14, 43, 37, 49, 21, 64, 77, 35, 44, 65, 56, 48, 81],  # 20, G2
                   [30, 32, 62, 10, 51, 39, 20, 7, 42, 17, 33, 19, 63, 45, 11, 29, 50, 52, 13, 69],  # 21, G1
                   [4, 1, 54, 9, 0, 22, 5, 6, 3, 12, 23, 15, 34, 26, 2, 27, 36, 47, 8, 41],       # 22, G0

                    [95, 91, 92, 61, 83, 87, 94, 24, 97, 76, 71, 38, 75, 99, 98, 93, 80, 96, 90, 89]   # 23, G4
                    ],

    "imagenet100": [list(range(100))]
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


from data_loader import iCIFAR100, ImageNet100
from torchvision import transforms

data_root = "../datasets"

def num_marco_classes_mapping(labels):

    macro_labels = [superClasses[classMap[label]] for label in labels]
    return len(list(set(macro_labels)))



data_function_mapping = {"cifar100_marco": iCIFAR100, "imagenet100": ImageNet100}

data_function_mapping_testing = {"cifar100_marco": iCIFAR100, "ImageNet100": ImageNet100}


mean_mapping = {"cifar100_marco": (0.4914, 0.4822, 0.4465), "imagenet100": (0.4914, 0.4822, 0.4465)}

std_mapping = {"cifar100_marco": (0.2023, 0.1994, 0.2010), "imagenet100": (0.2023, 0.1994, 0.2010)}


image_size_mapping = {"cifar100_marco": 32, "imagenet100": 224}


def label_to_dict(labels, outliers=False, cifar_marco_class=False):
    label_dict = dict()
    if cifar_marco_class:
        macro_labels = [superClasses[classMap[label]] for label in labels]
        marco_fine_map = {str(label): marco_label for label, marco_label in zip(labels, macro_labels)}
        marco_normalize_map = {str(marco_label): i for i, marco_label in enumerate(list(set(macro_labels)))}
    for i, l in enumerate(labels):
        if outliers is False:
            if cifar_marco_class:
                label_dict[str(l)] = marco_normalize_map[str(marco_fine_map[str(l)])]
            else:
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
                [#transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 #transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.ToTensor(),
                 normalize, ])  # normalize,

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(classes, cifar_marco_class=opt.marco_classes)
    train = True
    train_dataset = data_fun(root=data_root, train=train,
                             classes=classes, download=True,
                             transform=train_transform, label_dict=label_dict,
                             multiplier=opt.expand_data)

    print("dataset size", len(train_dataset))
    return train_dataset


def get_test_datasets(opt, class_idx = None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(classes, cifar_marco_class=opt.marco_classes)

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



