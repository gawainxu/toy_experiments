import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from Toy_model import toy_model, cnn
from Toy_train import label_mappings
from Toy_dataset import toy_dataset
import torchvision.transforms as transforms

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from plot_utils import plot_confusion_matrix

num_classes_mapping = {"E1": 2, "E2": 3,
                       "E3": 4, "E4": 4, "E5": 4,
                       "E6": 5, "E7": 5, "E8": 5,}

def parse_options():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_idx1', type=int, default=0)
    parser.add_argument('--data_idx2', type=int, default=-1)
    parser.add_argument('--data_idx3', type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="toy")
    parser.add_argument("--data_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="./toy_data_train")
    parser.add_argument("--test_data_path", type=str, default="./toy_data_test_inliers")

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="cnn", choices=["toy", "cnn", "vgg"])
    parser.add_argument("--experiment_name", type=str, default="E1")

    opt = parser.parse_args()
    return opt


def entropy(preds):
 
    preds = F.softmax(preds, dim=-1)
    logp = torch.log(preds + 1e-5)
    entropy = torch.sum(-preds * logp, dim=-1)

    return entropy

    
def osr_test(logits, mode="entropy"):

    if mode == "entropy":
        pred_ent = entropy(logits)
    
    return pred_ent


def feature_stats(inlier_features):

    stats = []
    for features in inlier_features:
        features = np.squeeze(np.array(features))
        mu = np.mean(features, axis=0)
        var = np.cov(features, rowvar=False)

        stats.append((mu, var))
    
    return stats


def knn(inlier_features, outlier_features, inlier_labels, k, mode="cosine"):

    inlier_features = np.concatenate(inlier_features, axis=0)
    inlier_features = np.squeeze(inlier_features)
    closest = []
    for features in outlier_features:
        if mode == "cosine":
            distances = np.matmul(features, inlier_features.T)
            distances = distances / np.linalg.norm(features, axis=1) / np.linalg.norm(inlier_features, axis=1)
            distances = np.abs(distances)
            distances = np.squeeze(distances)
        else:
            distances = inlier_features - features
            distances = np.linalg.norm(distances, axis=1)
        #idx = np.argpartition(distances, k)                # closest samples
        #idx = idx[:k]  
        idx = (-distances).argsort()[:k]                               
        labels = [inlier_labels[i] for i in idx]           # closest classes
        majority = np.argmax(np.bincount(labels))          # the majority
        #print(idx, labels, majority)
        closest.append(majority)

    return closest


def compare_hist(n1, n2, weights=False):

    n = np.sum(n1)
    normalized_n1 = n1*1.0 / n
    normalized_n2 = n2*1.0 / n

    distance = normalized_n1 - normalized_n2
    if weights == True:
        bins = len(n1)
        weights = np.arange(bins, 0, -1)
        weights = np.divide(weights, n *1.0)
        distance = np.multiply(distance, weights)

    distance = np.sum(np.abs(distance))
    return distance


def distances(stats, inlier_features, outlier_features, mode="mahalanobis"):

    dis_outliers = []
    closest_outliers = []
    dis_all = []
    for features in outlier_features:
        dis_min = 1e10
        idx = 0
        dis_ind = []
        for i, (mu, var) in enumerate(stats):
            #mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                dis =  np.matmul(features_normalized, np.linalg.inv(var))
                dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                dis = dis[0][0]
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            if dis_min > dis:
                dis_min = dis
                idx = i
            dis_ind.append(dis)
        dis_outliers.append(dis_min)
        closest_outliers.append(idx)
        dis_all.append(dis_ind)

    dis_inliers = np.empty([0])
    for c, features in enumerate(inlier_features):
        features = np.squeeze(np.array(features))
        mu, var = stats[c]                               # c
        centers = np.tile(mu, (len(features), 1))
        if mode == "mahalanobis":
            features_normalized = features - centers
            dis =  np.matmul(features_normalized, np.linalg.inv(var))
            dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
        else:
            dis = np.sum(np.abs(features - centers), axis=1)
        dis = np.diag(dis)
        dis_inliers =  np.concatenate((dis_inliers, dis), axis=0)

    return dis_inliers, dis_outliers, closest_outliers, dis_all                      # dis_all [num_features, num_classes]


def similarities(stats, inlier_features, outlier_features):

    s_outliers = []
    for features in outlier_features:
        features = np.squeeze(np.array(features))
        s_min = 1e10
        for mu, var in stats:
            similarities = np.matmul(features, mu.T)
            similarities = similarities / np.sum(np.abs(features)) / np.sum(np.abs(mu))
            similarities = np.abs(similarities)
            if s_min > similarities:
                s_min = similarities
        s_outliers.append(s_min)

    s_inliers = np.empty([0])
    for c, features in enumerate(inlier_features):
        features = np.squeeze(np.array(features))
        mu, var = stats[c]
        centers = np.tile(mu, (len(features), 1))
        similarities = np.matmul(features, centers.T)[:, 0]
        similarities = similarities / np.linalg.norm(mu) / np.linalg.norm(features, axis=1)
        similarities = np.abs(similarities)
        s_inliers =  np.concatenate((s_inliers, similarities), axis=0)

    return s_inliers, s_outliers


if __name__ == "__main__":

    batch_size = 1
    opt = parse_options()
    """
    # data_path1 and data_path2 can either be two tasks in continual learning or inliers and outliers in OSR
    data_path1 = "/home/zhi/projects/open_cross_entropy/toy_data_test_shapes"
    data_path2 = None #"/home/zhi/projects/open_cross_entropy/toy_data_test_outliers"
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])

    label_mapping1 = {"circle_black": 0, "rectangle_black": 1}  # {"circle_blue": 0, "rectangle_red": 1}  #
    label_mapping2 = {"rectangle_blue": 3, "rectangle_green": 4}
    dataset1 = toy_dataset(data_path1, label_mapping1, data_transform)

    if data_path2 is not None:
        dataset2 = toy_dataset(data_path2, label_mapping2, data_transform)
        dataset = ConcatDataset([dataset1, dataset2])
    else:
        dataset = dataset1
    """

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomHorizontalFlip(),
                                         ])
    if opt.data_idx1 > 0:
        label_mapping1 = label_mappings[opt.data_idx1]
        dataset = toy_dataset(opt.test_data_path, label_mapping1, data_transform)
    else:
        label_mapping1 = {}
        dataset = None

    if opt.data_idx2 > 0:
        label_mapping2 = label_mappings[opt.data_idx2]
        dataset2 = toy_dataset(opt.test_data_path, label_mapping2, data_transform)
        dataset = ConcatDataset([dataset, dataset2])
    else:
        label_mapping2 = {}
        dataset2 = None

    if opt.data_idx3 > 0:
        label_mapping3 = label_mappings[opt.data_idx3]
        dataset3 = toy_dataset(opt.test_data_path, label_mapping2, data_transform)
        dataset = ConcatDataset([dataset, dataset3])
    else:
        label_mapping3 = {}
        dataset3 = None

    data_loader = DataLoader(dataset, batch_size, num_workers=4, shuffle=False)


    num_classes = num_classes_mapping[opt.experiment_name]
    if "toy" in opt.model_name:
        model = toy_model(num_classes, in_channels=3, img_size=opt.data_size)
    elif "cnn" in opt.model_name:
        model = cnn(num_classes, in_channels=3, img_size=opt.data_size)
    model.load_state_dict(torch.load(opt.model_path))

    preds = []
    labels = []
    unequals = 0
    features = [[] for i in range(num_classes)]
    for i, (image, label) in enumerate(data_loader):
        image = image.float()
        label = label.numpy().item()
        #image = image.permute(0, 3, 1, 2)
        p = model(image)
        features[label].append(p.detach().numpy())
        p = torch.argmax(p)

        labels.append(label)
        preds.append(p.detach().numpy().item())
        if p.item() != label:
            unequals += 1

    acc = 1 - unequals * 1.0 / len(dataset)
    print("testing accuracy is ", acc)
    #conf_matrix = confusion_matrix(preds, labels)
    #plot_confusion_matrix(conf_matrix,"/home/zhi/projects/open_cross_entropy/confusion/test_E4.png")