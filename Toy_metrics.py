import argparse
import pickle
import numpy as np

from cka import linear_cka_gpt


def parse_options():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path1", type=str, default="/features/cifar10_resnet18_trail_0_128_0.005_train")
    parser.add_argument("--feature_path2", type=str, default="/features/cifar10_resnet18_trail_0_128_0.01_train")
    parser.add_argument("--feature_name", type=str, default="linear3")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--metric", type=str, default="cka")
    parser.add_argument("--if_data0", type=int, default=0)

    opt = parser.parse_args()

    if opt.feature_path1.split("/")[-1].split("_")[-2] == "0":
        opt.if_data0 = 1
    else:
        opt.if_data0 = 0

    return opt


if __name__ == "__main__":

    opt = parse_options()
    with open(opt.feature_path1, "rb") as f:
        features1, labels1 = pickle.load(f)

    with open(opt.feature_path2, "rb") as f:
        features2, labels2 = pickle.load(f)

    features1 = [f[opt.feature_name].numpy() for f in features1]
    features2 = [f[opt.feature_name].numpy() for f in features2]
    features1 = np.array(features1)
    features2 = np.array(features2)

    if opt.if_data0 == 1:
        indices1 = [i for i, l in enumerate(labels1) if l < 2]
        indices2 = [i for i, l in enumerate(labels2) if l < 2]
        features1 = features1[indices1]
        features2 = features2[indices2]

    if "cka" in opt.metric:
        print("cka is", linear_cka_gpt(features1, features2))