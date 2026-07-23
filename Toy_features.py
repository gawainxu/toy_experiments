import pickle
import os
import argparse
import torch
from torch.utils.data import DataLoader
from Toy_dataset import toy_dataset
from Toy_model import toy_model, cnn, toy_model_small
import torchvision.transforms as transforms

"""
label_mappings_full = [
                       [{"circle_blue": 0, "rectangle_red": 1},],  # E1,0

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}], # E2,1

                       [{"circle_blue": 0, "rectangle_red": 1},
                        {"circle_blue": 0, "rectangle_red": 1, "ellipse_red": 2, "rectangle_blue": 3},
                        {"circle_blue": 0, "rectangle_red": 1, "ellipse_red": 2, "rectangle_blue": 3, "triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E3,2

                       [{"circle_blue": 0, "rectangle_red": 1},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_yellow": 2, "rectangle_green": 3},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_yellow": 2, "rectangle_green": 3, "triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E4,3

                       [{"circle_blue": 0, "rectangle_red": 1},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_black": 2, "rectangle_black": 3},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_black": 2, "rectangle_black": 3,  "triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E5,4

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "ellipse_red": 3, "rectangle_blue": 4},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "ellipse_red": 3, "rectangle_blue": 4, "triangle_blue": 5, "ellipse_blue": 6, "triangle_red": 7}], # E6,5

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_yellow": 3, "rectangle_green": 4},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_yellow": 3, "rectangle_blue": 4, "triangle_blue": 5, "ellipse_blue": 6, "triangle_red":7}], # E7,6

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_black": 3, "rectangle_black": 4},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_black": 3, "rectangle_blue": 4, "triangle_blue": 5, "ellipse_blue": 6, "triangle_red":7}], # E8,7
                       ]

label_mappings_increment = [
                            [{"circle_blue": 0, "rectangle_red": 1}],  # E1,0

                            [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}], # E2,1

                            [{"circle_blue": 0, "rectangle_red": 1},
                             {"ellipse_red": 2, "rectangle_blue": 3},
                             {"triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E3,2

                             [{"circle_blue": 0, "rectangle_red": 1},
                              {"circle_yellow": 2, "rectangle_green": 3},
                              {"triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E4,3

                             [{"circle_blue": 0, "rectangle_red": 1},
                              {"circle_black": 2, "rectangle_black": 3},
                              {"triangle_blue": 4, "ellipse_blue": 5, "triangle_red": 6}], # E5,4

                             [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                              {"ellipse_red": 3, "rectangle_blue": 4},
                              {"triangle_blue": 5, "ellipse_blue": 6, "triangle_red": 7}], # E6,5

                             [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                              {"circle_yellow": 3, "rectangle_green": 4},
                              {"triangle_blue": 5, "ellipse_blue": 6, "triangle_red": 7}], # E7,6

                             [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                             {"circle_black": 3, "rectangle_black": 4},
                             {"triangle_blue": 5, "ellipse_blue": 6, "triangle_red": 7}], # E8,7
                             ]
"""

label_mappings_full = [
                       [{"circle_blue": 0, "rectangle_red": 1},],  # E1,0

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}], # E2,1

                       [{"circle_blue": 0, "rectangle_red": 1},
                        {"circle_blue": 0, "rectangle_red": 1, "ellipse_red": 2, "rectangle_blue": 3},
                        {"circle_blue": 0, "rectangle_red": 1, "ellipse_red": 2, "rectangle_blue": 3, "circle_black": 4, "rectangle_black": 5}], # E3,2

                       [{"circle_blue": 0, "rectangle_red": 1},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_black": 2, "rectangle_black": 3},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_black": 2, "rectangle_black": 3, "ellipse_red": 4, "triangle_blue": 5}], # E4,3

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "ellipse_red": 3, "rectangle_blue": 4},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "ellipse_red": 3, "rectangle_blue": 4, "circle_black": 5, "triangle_black": 6}], # E5,4

                       [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_black": 3, "rectangle_black": 4},
                        {"circle_blue": 0, "rectangle_red": 1, "circle_red": 2, "circle_black": 3, "rectangle_black": 4, "ellipse_red": 5, "triangle_blue": 6}], # E6,5

                       ]

label_mappings_increment = [
                            [{"circle_blue": 0, "rectangle_red": 1}],  # E1,0

                            [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2}], # E2,1

                            [{"circle_blue": 0, "rectangle_red": 1},
                             {"ellipse_red": 2, "rectangle_blue": 3},
                             {"circle_black": 4, "rectangle_black": 5}], # E3,2

                             [{"circle_blue": 0, "rectangle_red": 1},
                              {"circle_black": 2, "rectangle_black": 3},
                              {"ellipse_red": 4, "rectangle_blue": 5}],  # E4, 3

                             [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                              {"ellipse_red": 3, "rectangle_blue": 4},
                              {"circle_black": 5, "rectangle_black": 6}],   # E5, 4

                             [{"circle_blue": 0, "rectangle_red": 1, "circle_red": 2},
                             {"circle_black": 3, "rectangle_black": 4},
                             {"ellipse_red": 5, "rectangle_blue": 6}],   # E6, 5
                             ]

label_mappings_osr = [{"circle_red": 0},
                      {"rectangle_blue": 0},
                      {"rectangle_green": 0},
                      {"circle_green": 0},
                      {"ellipse_blue": 0},
                      {"ellipse_pink": 0}]


def parse_options():
    
    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--experiment_idx", type=int, default=1)
    parser.add_argument("--task_idx_model", type=int, default=0)
    parser.add_argument("--task_idx_data", type=int, default=0)
    parser.add_argument("--outliers_id", type=int, default=-1)    # >= 0 for outlier data
    parser.add_argument("--model_name", type=str, default="cnn", choices=["toy", "cnn", "vgg", "toy_small"])
    parser.add_argument("--model_path", type=str, default="./models/cnn_toy_E2.pth")
    parser.add_argument("--data_path", type=str, default="./toy_data_train")
    parser.add_argument("--data_size", type=int, default=64)
    parser.add_argument("--feature_save_path", type=str, default="./features/")
    parser.add_argument("--training_data", type=bool, default=True)

    opt = parser.parse_args()
    opt.num_classes = len(label_mappings_full[opt.experiment_idx][opt.task_idx_model])
    model_name = opt.model_path.split("/")[-1].split(".")[0]

    if opt.outliers_id >= 0:
        opt.label_mapping = label_mappings_osr[opt.outliers_id]
        class_name = list(label_mappings_osr[opt.outliers_id].keys())[0]
        opt.feature_save_path = opt.feature_save_path + model_name + "_" + class_name
        opt.data_path = "toy_data_test_outliers"
    elif opt.outliers_id == -1 and opt.training_data:
        opt.label_mapping = label_mappings_increment[opt.experiment_idx][opt.task_idx_data]
        class_name = list(label_mappings_increment[opt.experiment_idx][opt.task_idx_data].keys())
        opt.feature_save_path = opt.feature_save_path + model_name + "_task_" + str(opt.task_idx_model) + "_data_" + str(opt.task_idx_data) + "_train"
        opt.data_path = "toy_data_train"
    else:
        opt.label_mapping = label_mappings_increment[opt.experiment_idx][opt.task_idx_data]
        class_name = list(label_mappings_increment[opt.experiment_idx][opt.task_idx_data].keys())
        opt.feature_save_path = opt.feature_save_path + model_name + "_task_" + str(opt.task_idx_model) + "_data_" + str(opt.task_idx_data) + "_test"
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

    if opt.model_name == "toy":
        model = toy_model(len(opt.num_classes), in_channels=3, img_size=opt.data_size)
    if opt.model_name == "toy_small":
        model = toy_model_small(len(opt.num_classes), in_channels=3, img_size=opt.data_size)
    elif opt.model_name == "cnn":
        model = cnn(len(opt.num_classes), in_channels=3, img_size=opt.data_size)
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device("cpu")))
    model.eval()

    normalFeatureReading(model, opt)
    