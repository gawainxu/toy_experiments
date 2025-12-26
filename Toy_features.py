import pickle
import os
import argparse
import torch
from torch.utils.data import DataLoader
from Toy_dataset import toy_dataset
from Toy_model import toy_model, cnn
import torchvision.transforms as transforms

label_mappings = [{"circle_blue": 0, "rectangle_red": 1},    #E1
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}, #E2
                  {"circle_blue": 0, "rectangle_red": 1, "rectangle_blue": 2, "rectangle_green": 3}, #E3
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "rectangle_blue": 3, "rectangle_green": 4}, #E4
                  {"circle_blue": 0, "rectangle_red": 1, "circle_green": 2, "rectangle_green": 3}, #E5
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_green": 3, "rectangle_green": 4}, #E6
                  {"circle_blue": 0, "rectangle_red": 1, "rectangle_blue": 2, "rectangle_green": 3}, #E7
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_green": 3, "rectangle_green": 4}, #E8
                  {"circle_blue": 0, "rectangle_red": 1, "ellipse_blue": 2, "rectangle_blue": 3}, #E9
                  {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "ellipse_pink": 3, "rectangle_blue": 4}] #E10

label_mappings_osr = [{"circle_red": 0},
                      {"rectangle_blue": 0},
                      {"rectangle_green": 0},
                      {"circle_green": 0},
                      {"ellipse_blue": 0},
                      {"ellipse_pink": 0}]


def parse_options():
    
    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--inliers_id", type=int, default=1)
    parser.add_argument("--outliers_id", type=int, default=1)    # >= 0 for outlier data
    parser.add_argument("--model_name", type=str, default="cnn", choices=["toy", "cnn", "vgg"])
    parser.add_argument("--model_path", type=str, default= "./models/cnn_toy_E2.pth")
    parser.add_argument("--data_path", type=str, default="./toy_data_train")
    parser.add_argument("--data_size", type=int, default=64)
    parser.add_argument("--feature_save_path", type=str, default="./features/")
    parser.add_argument("--training_data", type=bool, default=False)

    opt = parser.parse_args()
    opt.num_classes = len(label_mappings[opt.inliers_id])
    model_name = opt.model_path.split("/")[-1].split(".")[0]

    if opt.outliers_id >= 0:
        opt.label_mapping = label_mappings_osr[opt.outliers_id]
        class_name = list(label_mappings_osr[opt.outliers_id].keys())[0]
        opt.feature_save_path = opt.feature_save_path + model_name + "_" + class_name
        opt.data_path = "toy_data_test_outliers"
    elif opt.outliers_id == -1 and opt.training_data:
        opt.label_mapping = label_mappings[opt.inliers_id]
        class_name = list(label_mappings[opt.inliers_id].keys())
        opt.feature_save_path = opt.feature_save_path + model_name + "_train"
        opt.data_path = "toy_data_train"
    else:
        opt.label_mapping = label_mappings[opt.inliers_id]
        class_name = list(label_mappings[opt.inliers_id].keys())
        opt.feature_save_path = opt.feature_save_path + model_name + "_test"
        opt.data_path = "toy_data_test_inliers"

    print(class_name)
    return opt
        

def normalFeatureReading(model, opt):
    
    outputs = []
    labels = []
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[1]
            print("hook working!!!", name, output.shape)
            activation[name] = output.detach()
        return hook
    
    # TODO loop through the layers and register hook to the specific layer
    # https://zhuanlan.zhihu.com/p/87853615

    for name, module in model.named_modules():
        print(name)
        module.register_forward_hook(get_activation(name))

    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         #transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
    dataset = toy_dataset(opt.data_path, opt.label_mapping, data_transform)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size = 1, 
                             num_workers=1, shuffle = False)

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        img = img.float()
        activation = {}
        hook_output = model(img)
        if type(hook_output) is tuple:
            hook_output = hook_output[1]
        outputs.append(activation)                            
        labels.append(label.numpy().item())

    with open(opt.feature_save_path, "wb") as f:
        pickle.dump((outputs, labels), f)


if __name__ == "__main__":

    opt = parse_options()

    if "toy" in opt.model_name:
        model = toy_model(num_classes=opt.num_classes, in_channels=3, img_size=opt.data_size)
    else:
        model = cnn(num_classes=opt.num_classes, in_channels=3, img_size=opt.data_size)
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device("cpu")))
    model.eval()

    normalFeatureReading(model, opt)
    