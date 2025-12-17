import torch
from Toy_dataset import toy_dataset
from Toy_model import toy_model_supcon
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle

from data_utils import TwoCropTransform
from losses import SupConLoss


if __name__ == "__main__":
    
    batch_size = 32
    out_dim = 10
    temp = 0.5
    epochs = 30
    lr = 1e-5
    data_path = "D://projects//open_cross_entropy//code//toy_data"
    test_data_path = "D://projects//open_cross_entropy//code//toy_data_test_inliers"
    model_path = "D://projects//open_cross_entropy//save//toy_model_supcon_E2"
    losses_path = "D://projects//open_cross_entropy//save//losses_model_supcon_E2"

    label_mapping = {"circleRed": 0, "rectangleRed": 1}               # , "circleGreen": 2
    num_classes = len(label_mapping)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])      # transforms.Normalize(mean=(0., 27.95993652, 30.653125), std=(0., 79.67449882, 82.92727418)) transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2),

    dataset = toy_dataset(data_path, label_mapping, TwoCropTransform(data_transform))
    data_loader = DataLoader(dataset, batch_size, num_workers=4, shuffle=True, drop_last=True)

    model = toy_model_supcon(out_dim)
    criteria = SupConLoss(temperature=temp)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_best = 1e10
    losses = []

    for e in range(epochs):
        loss_epoch = 0

        for i, (x, y) in enumerate(data_loader):
      
            x1, x2 = x[0], x[1]
            y = y.type(torch.LongTensor)
            #x1 = x1.permute(0, 3, 1, 2)
            #x2 = x2.permute(0, 3, 1, 2)
            x1 = x1.float()
            x2 = x2.float()

            images = torch.cat([x1, x2], dim=0)
            features, _ = model(images)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criteria(features, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        if loss_epoch < loss_best:
            #torch.save(model.state_dict(), model_path) 
            loss_best = loss_epoch
        
        losses.append(loss_epoch)
        print("epoch", e, "loss is", loss_epoch/len(dataset))

        
        model_path_epoch = model_path + "_" + str(e)
        torch.save(model.state_dict(), model_path_epoch)

    print("best loss: ", loss_best)
    with open(losses_path, "wb") as f:
        pickle.dump(losses, f)