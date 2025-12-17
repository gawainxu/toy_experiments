import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def parse_options():

    parser = argparse.ArgumentParser("Arguments")
    parser.add_argument("--feature_save_path", type=str, default="./features/E1_0")

    opt = parser.parse_args()
    return opt


def open_features(opt):

    with open(opt.feature_save_path, "rb") as f:
        features, labels = pickle.load(f)

    convs = [f["conv1"] for f in features]
    return convs, labels


def visualize_features(convs):

    for cf in convs:
        cf = torch.squeeze(cf)
        for ci in cf:
            ci = ci.numpy()
            ci = (ci-ci.min()) / (ci.max() - ci.min())
            plt.imshow(ci, vmin=0.0, vmax=1.0, cmap="gray")
            plt.show()



if __name__ == "__main__":

    opt = parse_options()
    convs, labels = open_features(opt)
    visualize_features(convs)