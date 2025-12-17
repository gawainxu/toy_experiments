import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from tornado.process import task_id

from Toy_model import renew_linear, toy_model
from  Toy_dataset import toy_dataset

"""
The script used to train and test linear probes for the new data using the
base models
"""

label_mappings = [{"circle_blue": 0, "rectangle_red": 1},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                  {"circle_blue": 0, "rectangle_red": 1, "rectangle_blue": 2, "rectangle_green": 3},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "rectangle_blue": 3, "rectangle_green": 4},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_green": 2, "rectangle_green": 3},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_green": 3, "rectangle_green": 4},

                  # task1 classes
                  {"rectangle_blue": 0, "rectangle_green": 1},
                  {"circle_green": 0, "rectangle_green": 1},
                  {"ellipse_blue": 0, "rectangle_blue": 1},
                  {"ellipse_pink": 0, "rectangle_blue": 1}]


def reset_linear(model, new_num_classes):

    input_dim = model.linear3.in_features
    new_out_linear = nn.Linear(input_dim, new_num_classes)
    model.linear3 = new_out_linear

    for name, param in model.named_parameters():
        if "linear3" not in name:
            param.requires_grad = False

    return model


if __name__ == "__main__":

    ori_model_path = "/home/zhi/projects/open_cross_entropy/models/toy_model_E2_99.pth"
    task_id = 1
    new_task_id = 8
    num_classes = len(label_mappings[task_id])
    new_num_classes = len(label_mappings[new_task_id])
    model = toy_model(num_classes)
    model.load_state_dict(torch.load(ori_model_path, map_location=torch.device("cpu")))
    model = reset_linear(model, new_num_classes)

    data_path = "/home/zhi/projects/open_cross_entropy/toy_data_train"
    data_path_test = "/home/zhi/projects/open_cross_entropy/toy_data_test_inliers"
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(), ])
    dataset = toy_dataset(data_path, label_mappings[new_task_id], data_transform)
    dataset_test = toy_dataset(data_path_test, label_mappings[new_task_id], data_transform)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=1, shuffle=True)


    lr = 1e-2
    epochs = 10
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    for e in range(epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
            x = x.float()
            y = y.type(torch.LongTensor)
            pred = model(x)
            loss = criteria(pred, y)
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        unequals = 0
        preds = []
        actuals = []
        accs = []
        for i, (x, y) in enumerate(test_data_loader):

            x = x.float()
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
