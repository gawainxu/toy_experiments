import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from toy_test import feature_stats, compare_hist
from view_weights import open_weights, find_removable
from Toy_model import toy_model, cnn


feature_keys = ['conv1', 'pooling', 'linear1', 'activation', 'linear2', 'linear3', '']


def parse_options():
    
    parser = argparse.ArgumentParser("Arguments")
    
    parser.add_argument("--model", type=str, default="toy", choices=["toy", "cnn"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="toy_toy_E2.pth")
    parser.add_argument("--inlier_features_path", type=str, default="toy_toy_E2_train")
    parser.add_argument("--outlier_features_path", type=str, default="toy_toy_E2_rectangle_blue")
    parser.add_argument("--data_size", type=int, default=64)
    parser.add_argument("--feature_to_view", type=str, default="linear2")
    parser.add_argument("--remove_weights", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="mahalanobis")
    
    opt = parser.parse_args()
    opt.current_dir = os.getcwd()
    
    opt.model_path = opt.current_dir + "/models/" + opt.model_name
        
    if opt.model == "toy":
        opt.feature_keys = ['conv1', 'pooling', 'linear1', 'linear2', 'linear3']
    elif opt.model == "cnn":
        opt.feature_keys = ['conv1', 'conv2', 'pooling', 'linear1', 'linear2', 'linear3']
        
    opt.inlier_features_path = opt.current_dir + "/features/" + opt.inlier_features_path
    opt.outlier_features_path = opt.current_dir + "/features/" + opt.outlier_features_path
    opt.hist_name = opt.current_dir + "/plots/hist_outlier_similarity_" + opt.model_name + ".png"
    
    return opt


def sort_features(features_in_dict, opt):
    
    if opt.model == "toy":
       sorted_features_in_dict = {'conv1': [], 'pooling': [], 'linear1': [], 'linear2': [], 'linear3': []}
    else:
       sorted_features_in_dict = {'conv1': [], 'conv2': [], 'pooling': [], 'linear1': [], 'linear2': [], 'linear3': []}
       
    for i, fd in enumerate(features_in_dict):
        
        #print(i, len(sorted_features_in_dict['linear3']))
        """
        for key in feature_keys:
            print(i, key, len(sorted_features_in_dict[key]))
            #print(sorted_features_in_dict["linear3"])
            print(fd[key])
            sorted_features_in_dict[key].append(fd[key])
        """
        sorted_features_in_dict["conv1"].append(np.squeeze(fd["conv1"].numpy()))
        if opt.model == "cnn":
            sorted_features_in_dict["conv2"].append(np.squeeze(fd["conv2"].numpy()))
        sorted_features_in_dict["pooling"].append(np.squeeze(fd["pooling"].numpy()))
        sorted_features_in_dict["linear1"].append(np.squeeze(fd["linear1"].numpy()))
        sorted_features_in_dict["linear2"].append(np.squeeze(fd["linear2"].numpy()))
        sorted_features_in_dict["linear3"].append(np.squeeze(fd["linear3"].numpy()))

    return sorted_features_in_dict


def sort_features_class(features_shuffled, labels, num_classes):

    features_sorted = [[] for _ in range(num_classes)]

    for f, l in zip(features_shuffled, labels):
        features_sorted[int(l)].append(f)

    return features_sorted


def distances(stats, inlier_features, outlier_features, mode="mahalanobis"):

    dis_outliers = []

    for features in outlier_features:
        dis_ind = []
        for i, (mu, var) in enumerate(stats):
            
            if mode == "mahalanobis":
                features_normalized = features - mu
                dis =  np.dot(features_normalized, np.linalg.inv(var))
                dis = np.dot(dis, features_normalized)
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            dis_ind.append(dis)          # distance of one point to one inlier class
        dis_outliers.append(dis_ind)


    dis_inliers = []
    for c, features in enumerate(inlier_features):
        features = np.squeeze(np.array(features))
        mu, var = stats[c]                               # c
        centers = np.tile(mu, (len(features), 1))
        if mode == "mahalanobis":
            features_normalized = features - centers
            dis =  np.dot(features_normalized, np.linalg.inv(var))
            dis = np.dot(dis, features_normalized.T)
            dis_c = np.diag(dis)
            dis_inliers.append(dis_c)
        else:
            dis = np.sum(np.abs(features - centers), axis=1)
            dis_inliers.append(dis)
                      # distance between pairs of points in one class
    return dis_inliers, dis_outliers


if __name__ == "__main__":

    opt = parse_options()
    
    if opt.model == 'toy':
        model = toy_model(opt.num_classes, in_channels=3, img_size=opt.data_size)
    else:
        model = cnn(opt.num_classes, in_channels=3, img_size=opt.data_size)
        
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device("cpu")))
    model.eval()

    with open(opt.inlier_features_path, "rb") as f:
        inlier_features, inlier_labels = pickle.load(f)

    with open(opt.outlier_features_path, "rb") as f:
        outlier_features, outlier_labels = pickle.load(f)

    #print(len(inlier_features))

    sorted_inlier_features_in_dict = sort_features(inlier_features, opt)
    sorted_outlier_features_in_dict = sort_features(outlier_features, opt)

    features_to_view_inliers = sorted_inlier_features_in_dict[opt.feature_to_view]
    features_to_view_outliers = sorted_outlier_features_in_dict[opt.feature_to_view]

    #inlier_labels = np.array(inlier_labels)
    #outlier_labels = np.array(outlier_labels)

    features_to_view_inliers = sort_features_class(features_to_view_inliers, inlier_labels, opt.num_classes)

    features_to_view_inliers = np.array(features_to_view_inliers)
    features_to_view_outliers = np.array(features_to_view_outliers)

    #print("features_to_view_inliers", features_to_view_inliers)
    #print("features_to_view_outliers", features_to_view_outliers)

    if opt.remove_weights:
        weights = open_weights(model, opt)
        remained = find_removable(weights)

        features_to_view_inliers = features_to_view_inliers[:,:,remained]
        features_to_view_outliers = features_to_view_outliers[:, remained]

    stats = feature_stats(features_to_view_inliers)

    dis_inliers, dis_outliers = distances(stats, features_to_view_inliers, features_to_view_outliers, opt.mode)

    for i, dis_c in enumerate(dis_inliers):
        print("Average Distance of Class ", i, "is", np.sum(dis_c)/dis_c.shape[0])

    outliers_to_class1 = []
    outliers_to_class2 = []
    for dis_ind in dis_outliers:
        outliers_to_class1.append(dis_ind[0])
        outliers_to_class2.append(dis_ind[1])

    print("Outliers to Class0", sum(outliers_to_class1)/len(outliers_to_class1))
    print("Outliers to Class1", sum(outliers_to_class2)/len(outliers_to_class2))

    n_inlier, _, _ = plt.hist(outliers_to_class1, bins=200, range=(0, 2000), alpha=0.5, label="Outlier Similarities1")
    #n_outlier, _, _ = plt.hist(outliers_to_class2, bins=200, range=(0, 2000), alpha=0.5, label="Outlier Similarities2")
    n_outlier, _, _ = plt.hist(outliers_to_class2[0] + outliers_to_class2[1], bins=200, range=(0, 2000), alpha=0.5, label="Inlier Similarities2")
    plt.xlabel('Distances')
    plt.ylabel('Counts')
    plt.legend(prop ={'size': 10})
    plt.title('Histogram of Mahalanobis Distances', fontsize=15)
    print("Histogram Distance", compare_hist(n_inlier, n_outlier))
    print("Histogram Distance (weights)", compare_hist(n_inlier, n_outlier, weights=True))
    plt.savefig(opt.hist_name)           
    
    
"""
when the outlier similarity to class 1 and class 2 becomes more different and outlier is more similar to class 2 as inliers, 
it means the color feature are discarded

For Supcon, if remove small weights in head, if the above happens, it means head helps to remove shared features
"""