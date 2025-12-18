import torch
import torch.nn as nn


class toy_model(nn.Module):

    def __init__(self, num_classes, in_channels=3, img_size=32) -> None:
        super(toy_model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        #self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.linear1 = nn.Linear(int(img_size / 2) * int(img_size / 2) * 10, 1000)
        self.linear2 = nn.Linear(1000, 20)
        self.linear3 = nn.Linear(20, num_classes)
        self.activation = nn.ReLU()


    def forward(self, x):

        y = self.conv1(x)
        y = self.pooling(y)
        #y = self.conv2(y)
        #y = self.pooling(y)
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)

        return y


class cnn(nn.Module):

    def __init__(self, num_classes, in_channels=3, img_size=32) -> None:
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.linear1 = nn.Linear(int(img_size / 4) * int(img_size / 4) * 10, 1000)
        self.linear2 = nn.Linear(1000, 20)
        self.linear3 = nn.Linear(20, num_classes)
        self.activation = nn.ReLU()


    def forward(self, x):

        y = self.conv1(x)
        y = self.pooling(y)
        y = self.conv2(y)
        y = self.pooling(y)
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)

        return y



class toy_model_supcon(nn.Module):

    def __init__(self, out_dim=20) -> None:
        super(toy_model_supcon, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        #self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.linear1 = nn.Linear(32*32*10, 1000)
        self.linear2 = nn.Linear(1000, out_dim)

        self.head = nn.Linear(out_dim, 10)
        self.activation = nn.ReLU()

    def forward(self, x):

        y = self.conv1(x)
        y = self.pooling(y)
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)

        out = self.head(y)

        return out, y


def init_weights(m: nn.Module, mode="kaiming"):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        if mode == "kaiming":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif mode == "zeros":
            nn.init.zeros_(m.weight)
        elif mode == "ones":
            nn.init.ones_(m.weight)
        elif mode == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight)
        elif mode == "xavier_normal":
            nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].zero_()


def updata_model(model, new_num_classes):

    old_num_clases = model.linear3.out_features
    input_dim = model.linear3.in_features
    new_out_linear = nn.Linear(input_dim, old_num_clases+new_num_classes)

    with torch.no_grad():
         new_out_linear.weight[:old_num_clases, :] = model.linear3.weight
         model.linear3 = new_out_linear

    return  model


def renew_linear(model):

    num_clases = model.linear3.out_features
    input_dim = model.linear3.in_features

    new_out_linear = nn.Linear(input_dim, num_clases)
    model.linear3 = new_out_linear

    for name, param in model.named_parameters():
        if "linear3" not in name:
            param.requires_grad = False

    return model


if __name__ == "__main__":

    model = toy_model(num_classes=10)
    model.apply(init_weights)
    print(model.linear3.weight.shape)



        