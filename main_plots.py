import argparse

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss_file_paths', type=str, nargs='*', default='')
    parser.add_argument('--plot_save_path', type=str, default='')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_arguments()
    base_names = ["B0", "B1", "B2",
                  "B3", "B4", "B5", "B6"]

    all_accs = {}
    all_losses = {}

    for i, loss_file_path in enumerate(opt.loss_file_paths):
        with open(loss_file_path, 'rb') as f:
            loss_file_path = os.path.join("./save/CE/cifar100_marco_models", loss_file_path)
            losses, accs = pickle.load(f)
            all_losses[base_names[i]] = losses
            all_accs[base_names[i]] = accs


    for name, accs in all_accs.items():
        plt.plot(accs, label=name)

    plt.legend()
    plt.savefig(os.path.join(opt.plot_save_path, "accs.pdf"))


