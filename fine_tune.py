import os
import argparse

import torch
import torchvision.transforms as transforms
from torch.optim import SGD
import numpy as np

from Toy_model import toy_model, cnn
from Toy_dataset import toy_dataset
from torch.utils.data import DataLoader


def parse_options():

    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--dataset", type=str, default="toy_shape")
    parser.add_argument("--data_path", type=str, default="./toy_data_train_shapes")
    parser.add_argument("--test_data_path", type=str, default="./toy_data_test_shapes")
    parser.add_argument("--data_size", type=int, default=64)

    parser.add_argument("--model_name", type=str, default="toy", choices=["toy", "cnn", "vgg"])
    parser.add_argument("--model_path", type=str, default="./models/")
    parser.add_argument("--last_model_path", type=str, default="./models/toy_model_E1_99.pth")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--freeze_layers", type=list, default=["conv1", "conv2", "linear1", "linear2"])

    opt = parser.parse_args()
    opt.label_mapping = {"circle_black": 0, "rectangle_black": 1}
    model_name = opt.model_name + "_" + opt.dataset + ".pth"
    opt.model_path = os.path.join("./models/", model_name)

    return opt


if __name__ == "__main__":

    opt = parse_options()
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])
    dataset = toy_dataset(opt.data_path, opt.label_mapping, data_transform)
    data_loader = DataLoader(dataset, opt.batch_size, num_workers=1, shuffle=True)
    dataset_test = toy_dataset(opt.test_data_path, opt.label_mapping, data_transform)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=1, shuffle=True)

    model = toy_model(num_classes=opt.num_classes, in_channels=3, img_size=opt.data_size)
    model.load_state_dict(torch.load(opt.last_model_path, weights_only=True))

    model.train()
    model = model.cuda()

    for name, param in model.named_parameters():
        for l in opt.freeze_layers:
            if l in name:
                print(name)
                param.requires_grad = False

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=opt.lr)

    loss_best = 1e10
    acc_best = -1e10
    losses = []
    accs = []

    for e in range(opt.epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
            x = x.float()
            x = x.cuda()
            y = y.type(torch.LongTensor)
            y = y.cuda()
            pred = model(x)
            loss = criteria(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        if loss_epoch < loss_best:
            loss_best = loss_epoch

        losses.append(loss_epoch)
        print("epoch", e, "loss is", loss_epoch / len(dataset))
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
            # print(pred, y)
            preds.append(pred.item())
            actuals.append(y.item())
            if pred.item() != y.item():
                unequals += 1

        acc = 1 - unequals * 1.0 / len(dataset_test)
        print("testing accuracy is ", acc)
        accs.append(acc)
        if acc > acc_best:
            acc_best = acc

    print("Avg Acc", sum(accs) / len(accs))
    print("Best Acc", acc_best)




