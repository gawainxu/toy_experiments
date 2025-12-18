import torch
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from Toy_model import renew_linear, toy_model
from  Toy_dataset import toy_dataset


label_mappings = [{"circle_blue": 0, "rectangle_red": 1},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                  {"circle_blue": 0, "rectangle_red": 1, "rectangle_blue": 2, "rectangle_green": 3},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "rectangle_blue": 3, "rectangle_green": 4},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_green": 2, "rectangle_green": 3},
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_green": 3, "rectangle_green": 4},

                  {"rectangle_blue": 2, "rectangle_green": 3},
                  {"circle_red": 2, "rectangle_blue": 3, "rectangle_green": 4},
                  {"circle_green": 2, "rectangle_green": 3},
                  {"circle_red": 2, "circle_green": 3, "rectangle_green": 4},]


if __name__ == "__main__":

    """
    """

    ori_model_path = "/home/zhi/projects/open_cross_entropy/models/toy_model_E1_99.pth"
    data_path = "/home/zhi/projects/open_cross_entropy/toy_data_train"
    data_path_test = "/home/zhi/projects/open_cross_entropy/toy_data_test_inliers"
    num_classes = 5
    exp_id = 3
    exp_id_test = 1

    model = toy_model(num_classes)
    model.load_state_dict(torch.load(ori_model_path, map_location=torch.device("cpu")))
    model = renew_linear(model)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),])
    dataset = toy_dataset(data_path, label_mappings[exp_id], data_transform)
    dataset_test = toy_dataset(data_path_test, label_mappings[exp_id_test], data_transform)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=True)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=1, shuffle=True)

    lr = 1e-3
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


