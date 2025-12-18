import torch
from Toy_model import toy_model, updata_model, init_weights
from Toy_dataset import toy_dataset, continual_buffer, iCIFAR100, mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import pickle
from sklearn.metrics import confusion_matrix
import os
import argparse
from plot_utils import plot_confusion_matrix


label_mappings = [{"circle_blue": 0, "rectangle_red": 1},  # E1
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}, # E2
                  {"rectangle_blue": 2, "rectangle_green": 3}, # E3
                  {"rectangle_blue": 3, "rectangle_green": 4}, # E4
                  {"circle_green": 2, "rectangle_green": 3}, # E5
                  {"circle_green": 3, "rectangle_green": 4}, # E6

                  {"ellipse_blue": 2, "rectangle_blue": 3}, # E7
                  {"ellipse_blue": 3, "rectangle_blue": 4}, # E8
                  {"ellipse_pink": 2, "rectangle_blue": 3}, # E9
                  {"ellipse_pink": 3, "rectangle_blue": 4}] # E10


def parse_options():

    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data_root", type=str, default="../datasets")
    parser.add_argument("--data_path", type=str, default= "./toy_data_train")
    parser.add_argument("--test_data_path", type=str, default="./toy_data_test_inliers")
    parser.add_argument("--data_size", type=int, default=32)
    parser.add_argument("--classes", type=list, default=list(range(10)))

    parser.add_argument("--model_name", type=str, default="toy", choices=["toy", "cnn", "vgg"])
    parser.add_argument("--model_path", type=str, default="./models/toy_model_E1_kaiming")
    parser.add_argument("--losses_path", type=str, default="./losses_model_E1_kaiming")
    parser.add_argument("--last_model_path", type=str, default=None)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_options()
    last_label_mapping = label_mappings[0]

    label_mapping = label_mappings[1]

    if "mnist" in opt.dataset:
        data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        dataset = mnist(opt.data_root, classes=opt.classes, transform=data_transform)
        dataset_test = mnist(opt.data_root, transform=data_transform)
        in_channels = 1
    elif "cifar" in opt.dataset:
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean= (0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
        dataset = iCIFAR100(opt.data_root, classes=opt.classes, transform=data_transform)
        dataset_test = iCIFAR100(opt.data_root, classes=opt.classes, transform=data_transform)
        in_channels = 3
    else:
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomHorizontalFlip(),
                                             ])
        dataset = toy_dataset(opt.data_path, label_mapping, data_transform)
        dataset_test = toy_dataset(opt.test_data_path, label_mapping, data_transform)
        in_channels = 3
        opt.classes = [label_mapping[k] for k in opt.label_mapping.keys()]

    data_loader = DataLoader(dataset, opt.batch_size, num_workers=1, shuffle=True)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=1, shuffle=True)

    if opt.last_model_path is not None:
        model = toy_model(len(last_label_mapping), in_channels=in_channels)
        model.load_state_dict(torch.load(opt.last_model_path, weights_only=True))
        updata_model(model, new_num_classes=len(label_mapping))
        if opt.buffer_size > 0:
            old_dataset = toy_dataset(opt.data_path, last_label_mapping, data_transform)
            old_dataset = continual_buffer(old_dataset, opt.buffer_size)
            dataset = torch.utils.data.ConcatDataset([dataset, old_dataset])
    else:
        model = toy_model(len(opt.classes), in_channels=in_channels, img_size=opt.data_size)
        model.apply(init_weights)

    model.train()
    model = model.cuda()

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=opt.lr)

    loss_best = 1e10
    acc_best = -1e10
    losses = []
    accs = []
    confusions = []
    for e in range(opt.epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
      
            x = x.float()
            x = x.cuda()
            y = y.type(torch.LongTensor)
            y = y.cuda()
            pred = model(x)
            loss = criteria(pred, y)
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        if loss_epoch < loss_best:
            #torch.save(model.state_dict(), model_path) 
            loss_best = loss_epoch
        
        losses.append(loss_epoch)
        print("epoch", e, "loss is", loss_epoch/len(dataset))
        loss_epoch = 0

        unequals = 0
        preds = []
        actuals = []
        for i, (x, y) in enumerate(test_data_loader):

            x = x.float()
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            pred = torch.argmax(pred)
            #print(pred, y)
            preds.append(pred.item())
            actuals.append(y.item())
            if pred.item() != y.item():
                unequals += 1

        #conf_matrix = confusion_matrix(preds, actuals)
        #confusions.append(conf_matrix)
        #plot_confusion_matrix(conf_matrix, "D://projects//open_cross_entropy//save//confusion_class3_" + str(e) + ".png")

        acc = 1-unequals*1.0 / len(dataset_test)
        print("testing accuracy is ", acc)
        accs.append(acc)
        if acc > acc_best:
            #torch.save(model.state_dict(), model_path) 
            acc_best = acc
        
        model_path_epoch = opt.model_path + "_" + str(e) + ".pth"
        torch.save(model.state_dict(), model_path_epoch)

    print("best loss: ", loss_best/len(dataset), "best acc: ", acc_best)
    with open(opt.losses_path, "wb") as f:
        pickle.dump((losses, accs), f)