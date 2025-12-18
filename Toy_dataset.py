import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR100, MNIST
from PIL import Image


class toy_dataset(Dataset):

    def __init__(self, data_path, label_map, transform=None):

        self.images = []
        self.labels = []
        self.transform = transform
        #os.chdir(data_path)

        for f in os.listdir(data_path):
            img = cv2.imread(data_path + "/" + f)
            label = f.split("_")[0] + "_" + f.split("_")[1]
            if label not in label_map.keys():
                continue
            self.images.append(img)
            label = label_map[label]
            self.labels.append(label)

    def __getitem__(self, index):

        image = self.images[index]
        image = self.transform(image)
        #image = (image - 128.) / 128.
        return image, self.labels[index]

    def __len__(self):

        return len(self.images)


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 classes=range(100),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 label_dict=None):
        super(iCIFAR100, self).__init__(root,
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


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class mnist(MNIST):

    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 label_dict=None):
        super(mnist, self).__init__(root, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)

        self.label_dict = label_dict
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data)):  # subsample
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])  # it is torch tensor !!!!!!!!!!!!!

            print(len(test_data))
            self.testdata = torch.stack(test_data).numpy()
            self.testlabels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.traindata[index], self.trainlabels[index]
        else:
            img, target = self.testdata[index], self.testlabels[index]

        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target.item())]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)

    def get_image_class(self, label):
        return self.traindata[np.array(self.trainlabels) == label]



def continual_buffer(dataset, buffer_size):

    lengths = [len(dataset), buffer_size]
    _, dataset = torch.utils.data.dataset.random_split(dataset, lengths)

    return dataset


if __name__ == "__main__":
    """
    data_path = "D://projects//open_cross_entropy//code//toy_data"
    label_mapping = {"circleGreen": 0, "rectangleRed": 1, "circleRed": 2} 
    data = toy_dataset(data_path, label_mapping)

    images = np.array(data.images)
    images = np.reshape(images, (-1, 3))
    print(np.mean(images, axis=0))
    print(np.std(images, axis=0))
    print(np.max(images, axis=0))
    print(np.min(images, axis=0))
    """

    data_root = "../datasets"
    dataset = mnist(root=data_root, classes=[0,1,2])
    img, l = dataset[0]
