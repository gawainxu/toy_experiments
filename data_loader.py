from torchvision.datasets import CIFAR10, CIFAR100
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import dataUtil
from dataUtil import osr_splits_inliers
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class iCIFAR10(CIFAR10):
    def __init__(self, root, classes=range(10), train=True, transform=None,
                 target_transform=None, download=False, label_dict=None):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.label_dict = label_dict

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

            print("Final Data Size ", len(self.train_data))

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 classes=range(100),
                 superClass = None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 label_dict = None,
                 multiplier = 1):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.label_dict = label_dict
        self.multiplier = multiplier

        if superClass is not None:
            classes = [dataUtil.classMap[n] for n in dataUtil.superClasses[superClass]] 

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels
            
        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):

        if self.train:
            if self.multiplier > 1:
                index = index % len(self.train_data)
            img, target = self.train_data[index], self.train_labels[index]
        else:
            if self.multiplier > 1:
                index = index % len(self.test_data)
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return int(len(self.train_data) * self.multiplier)
        else:
            return int(len(self.test_data) * self.multiplier)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels        


class ImageNet100(Dataset):

    def __init__(self, root, classes=range(100), train=True, opt=None, transform=None, multiplier = 1,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        if train:
            data_path = root + "/imagenet100_train"
        else:
            data_path = root + "/imagenet100_test"

        dataset = SelectImageFolder(data_path, classes)
        self.images = []
        self.labels = []
        self.transform = transform

        for img, l in dataset:
            self.images.append(img)
            self.labels.append(l)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):

        return len(self.images)


class SelectImageFolder(ImageFolder):
    def __init__(self, root, classes=range(100), **kwargs):

        self.target_class_indices = classes
        super().__init__(root, **kwargs)

    def find_classes(self, directory):

        classes_all = [d.name for i, d in enumerate(os.scandir(directory)) if d.is_dir()]
        classes_all.sort()

        classes = [n for i, n in enumerate(classes_all)
                            if i in self.target_class_indices]
        classes_to_idx = {cls_name: i for cls_name, i in zip(classes, self.target_class_indices)}

        return classes, classes_to_idx



if __name__ == "__main__":
    transform = transforms.Compose([
       # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])                                      # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    classes = osr_splits_inliers["cifar100_marco"][7]
    root_path = "../datasets"
    label_dict = dataUtil.label_to_dict(osr_splits_inliers["cifar100_marco"][7], cifar_marco_class=True)
    dataset = iCIFAR100(root='../datasets', classes=classes, transform=None, label_dict=label_dict)
    print(len(dataset))
    img, l = dataset[0]
    img = Image.fromarray(img)
    img.save("95.png")