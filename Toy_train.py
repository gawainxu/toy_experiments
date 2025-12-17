import torch
from Toy_model import toy_model, updata_model, init_weights
from Toy_dataset import toy_dataset, continual_buffer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
import pickle
from sklearn.metrics import confusion_matrix
import os
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


if __name__ == "__main__":
    
    batch_size = 32
    epochs = 100
    lr = 1e-3
    buffer_size = 0

    data_path = "/home/zhi/projects/open_cross_entropy/toy_data_train"
    test_data_path = "/home/zhi/projects/open_cross_entropy/toy_data_test_inliers"
    model_path = "/home/zhi/projects/open_cross_entropy/models/toy_model_E1_kaiming"
    losses_path = "/home/zhi/projects/open_cross_entropy/losses_model_E1_kaiming"
    # for continual mode
    last_model_path = None #"/home/zhi/projects/open_cross_entropy/models/toy_model_E2_99.pth"
    last_label_mapping = label_mappings[0]


    label_mapping = label_mappings[1]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])      # transforms.Normalize(mean=(0., 27.95993652, 30.653125), std=(0., 79.67449882, 82.92727418))

    dataset = toy_dataset(data_path, label_mapping, data_transform)
    dataset_test = toy_dataset(test_data_path, label_mapping, data_transform)
    data_loader = DataLoader(dataset, batch_size, num_workers=1, shuffle=True)
    test_data_loader = DataLoader(dataset_test, 1, num_workers=1, shuffle=True)

    if last_model_path is not None:
        model = toy_model(len(last_label_mapping))
        model.load_state_dict(torch.load(last_model_path, weights_only=True))
        updata_model(model, new_num_classes=len(label_mapping))
        if buffer_size > 0:
            old_dataset = toy_dataset(data_path, last_label_mapping, data_transform)
            old_dataset = continual_buffer(old_dataset, buffer_size)
            dataset = torch.utils.data.ConcatDataset([dataset, old_dataset])
    else:
        model = toy_model(len(label_mapping))
        model.apply(init_weights)

    model.train()

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    loss_best = 1e10
    acc_best = -1e10
    losses = []
    accs = []
    confusions = []
    for e in range(epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
      
            x = x.float()
            y = y.type(torch.LongTensor)
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
            pred = model(x)
            pred = torch.argmax(pred)
            #print(pred, y)
            preds.append(pred.item())
            actuals.append(y.item())
            if pred.item() != y.item():
                unequals += 1

        conf_matrix = confusion_matrix(preds, actuals)
        confusions.append(conf_matrix)
        #plot_confusion_matrix(conf_matrix, "D://projects//open_cross_entropy//save//confusion_class3_" + str(e) + ".png")

        acc = 1-unequals*1.0 / len(dataset_test)
        print("testing accuracy is ", acc)
        accs.append(acc)
        if acc > acc_best:
            #torch.save(model.state_dict(), model_path) 
            acc_best = acc
        
        model_path_epoch = model_path + "_" + str(e) + ".pth"
        torch.save(model.state_dict(), model_path_epoch)

    print("best loss: ", loss_best/len(dataset), "best acc: ", acc_best)
    with open(losses_path, "wb") as f:
        pickle.dump((losses, accs, confusions), f)