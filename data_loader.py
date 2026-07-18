from torchvision.datasets import CIFAR10, CIFAR100
import dataUtil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



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
                 label_dict = None):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.label_dict = label_dict

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



if __name__ == "__main__":
    transform = transforms.Compose([
       # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])                                      # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    # train_set = iCIFAR100(root='../datasets/', train=True,
    #                        classes=range(0, 10),
    #                        download=False, transform=None)
    # train_set = apply_transform(train_set, TwoCropTransform(transform))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True, num_workers=1)
    # for i, (img, l) in enumerate(train_loader):
    #     if i == 0:
    #         break
    print("testing")
    root_path = "../datasets"
    dataset = iCIFAR100(root='../datasets', classes=[0])
    print(dataset[0][1])
    print(len(dataset))