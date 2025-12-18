import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def rescale_matrix(orig_matrix, scale):

    orig_size = orig_matrix.shape[0]
    target_matrix = np.zeros([scale*orig_size, scale*orig_size])

    for i in range(orig_size):
        for j in range(orig_size):
            target_matrix[i*scale : (i+1)*scale, j*scale : (j+1)*scale] = orig_matrix[i, j]
    
    return target_matrix


def plot_confusion_matrix(conf_matrix, save_path):

    resized_matrix = rescale_matrix(conf_matrix, scale=20)

    plt.imshow(resized_matrix, cmap="Blues")
    plt.imsave(save_path, resized_matrix, cmap="Blues")


def precision_recall_accuracy(conf_matrix):

    num_classes = conf_matrix.shape[0]
    P = []
    R = []

    for c in range(num_classes):
        P.append(conf_matrix[c, c] * 1.0 / (np.sum(conf_matrix[:, c]) + 1e-6))
        R.append(conf_matrix[c, c] * 1.0 / (np.sum(conf_matrix[c, :]) + 1e-6))

    precision = sum(P) / num_classes
    recall = sum(R) / num_classes
    accuracy = np.trace(conf_matrix) * 1.0 / np.sum(conf_matrix)

    return precision, recall, accuracy


