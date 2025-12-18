import torch

import numpy as np
from Toy_model import toy_model


def open_weights(model, opt):

    if opt.loss == "ce":
        weight_name = "linear3.weight"
    else:
        weight_name = "head.weight"
        
    for name, param in model.named_parameters():
        if name == weight_name:
            return param.data


def find_removable(weights):

    w0 = weights[0].numpy()
    w1 = weights[1].numpy()

    w0_mean = np.mean(np.abs(w0))
    w1_mean = np.mean(np.abs(w1))

    removable_index = []
    for i, (p, q) in enumerate(zip(w0, w1)):
        if np.abs(p) < w0_mean and np.abs(q) < w1_mean:
            removable_index.append(i)

    remained = [i for i in range(20) if i not in removable_index]     #  

    return remained

    
if __name__ == "__main__":
    
    num_classes = 2
    model_path = "D://projects//open_cross_entropy//save//toy_model_E2_199"
    model = toy_model(num_classes)

    weights = open_weights(model)

    removable_index = find_removable(weights)
    print(removable_index)