import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.functional import auroc

from Toy_distances import mahalanobis_distances

from Toy_model import toy_model, cnn
from Toy_dataset import toy_dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse
import pickle


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

    parser.add_argument("--dataset", type=str, default="toy_shape")
    parser.add_argument("--data_path_train", type=str, default="./toy_data_train")
    parser.add_argument("--data_path_inliers", type=str, default="./toy_data_test_inliers")
    parser.add_argument("--data_path_outliers", type=str, default="./toy_data_test_outliers")
    parser.add_argument("--data_size", type=int, default=64)
    parser.add_argument("--inliers_id", type=int, default=1)
    parser.add_argument("--outliers_id", type=int, default=1)

    parser.add_argument("--model_type", type=str, default="cnn", choices=["toy", "cnn", "vgg"])
    parser.add_argument("--model_path", type=str, default="./models/cnn_toy_E2.pth")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--train_feature_path", type=str, default="./features/cnn_toy_E2_train")
    parser.add_argument("--inliers_feature_path", type=str, default="./features/cnn_toy_E2_test")
    parser.add_argument("--outliers_feature_path", type=str, default="./features/cnn_toy_E2_rectangle_blue")
    parser.add_argument("--feature_to_use", type=str, default="linear2")

    opt = parser.parse_args()

    opt.inliers_mapping = label_mappings[opt.inliers_id]
    opt.outliers_mapping = label_mappings_osr[opt.outliers_id]

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
        # idx = np.argpartition(distances, k)                # closest samples
        # idx = idx[:k]
        idx = (-distances).argsort()[:k]
        labels = [inlier_labels[i] for i in idx]  # closest classes
        majority = np.argmax(np.bincount(labels))  # the majority
        # print(idx, labels, majority)
        closest.append(majority)

    return closest


def seperate_class(features, labels, num_classes):

    seperated_features = [[] for i in range(num_classes)]
    for i, l in enumerate(labels):
        seperated_features[l].append(features[i])

    return seperated_features


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
        s_inliers = np.concatenate((s_inliers, similarities), axis=0)

    return s_inliers, s_outliers


def AUROC(labels, probs):
    '''
    ROC:
        X: False positive rate
        Y: True positive rate
    '''
    fpr, tpr, threholds = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)

    # plot the AUROC curve
    # plt.plot([0, 1], [0, 1], linestyle="--")
    # plt.plot(fpr, tpr)
    # plt.ylabel("TPR (Sensitity)")
    # plt.xlabel("FPR (1 - Specificity)")

    # plt.savefig(opt.auroc_save_path)

    return auroc


def OSCR(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


if __name__ == "__main__":

    opt = parse_options()
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         #transforms.RandomHorizontalFlip(),
                                         ])
    train_dataset = toy_dataset(opt.data_path_train, opt.inliers_mapping, data_transform)
    inliers_dataset = toy_dataset(opt.data_path_inliers, opt.inliers_mapping, data_transform)
    outliers_dataset = toy_dataset(opt.data_path_outliers, opt.outliers_mapping, data_transform)

    train_data_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
    inliers_data_loader = DataLoader(inliers_dataset, batch_size=1, num_workers=4, shuffle=False)
    outliers_data_loader = DataLoader(outliers_dataset, batch_size=1, num_workers=4, shuffle=False)


    if "toy" in opt.model_type:
        model = toy_model(opt.num_classes, in_channels=3, img_size=64)
    else:
        model = cnn(opt.num_classes, in_channels=3, img_size=64)
    model.load_state_dict(torch.load(opt.model_path))

    ################################### msp / energy #########################################
    m = torch.nn.Softmax(dim=1)
    preds_inliers = []
    probs_inliers = []
    labels_inliers = []
    unequals = 0
    for i, (image, label) in enumerate(inliers_data_loader):
        image = image.float()
        label = label.numpy().item()
        pred = model(image)
        pred = m(pred)
        prob = torch.max(pred)
        pred = torch.argmax(pred)

        labels_inliers.append(label)
        preds_inliers.append(pred.detach().numpy().item())
        probs_inliers.append(prob.detach().numpy().item())
        if pred.item() != label:
            unequals += 1

    #print("preds_inliers", preds_inliers)
    #print("labels_inliers", labels_inliers)
    #acc = 1 - unequals * 1.0 / len(inliers_dataset)
    #print("testing accuracy is ", acc)
    print("prob inliers", probs_inliers)

    preds_outliers = []
    probs_outliers = []
    labels_outliers = []
    unequals = 0
    for i, (image, label) in enumerate(outliers_data_loader):
        image = image.float()
        label = label.numpy().item()
        pred = model(image)
        pred = m(pred)
        prob = torch.max(pred)
        pred = torch.argmax(pred)

        labels_outliers.append(label)
        preds_outliers.append(pred.detach().numpy().item())
        probs_outliers.append(prob.detach().numpy().item())
        if pred.item() != label:
            unequals += 1

    #print("preds_outliers", preds_outliers)
    print("prob outliers", probs_outliers)
    print(sum(probs_outliers)/len(probs_outliers))

    binary_labels = [1 for _ in range(len(labels_inliers))] + [0 for _ in range(len(labels_outliers))]
    binary_labels = np.array(binary_labels)
    probs_binary = probs_inliers + probs_outliers
    #probs_binary = [-i for i in probs_binary]
    probs_binary = np.array(probs_binary)
    auroc_msp = AUROC(binary_labels, probs_binary)
    print("auroc_msp", auroc_msp)

    ################################ Mahalanobis ###############################################
    with open(opt.train_feature_path, "rb") as f:
        train_features, train_labels = pickle.load(f)
    train_features = [train_feature[opt.feature_to_use].detach().numpy() for train_feature in train_features]
    train_features = np.squeeze(np.array(train_features))
    train_features_sorted = seperate_class(train_features, train_labels, opt.num_classes)

    with open(opt.inliers_feature_path, "rb") as f:
        inlier_features, inlier_labels = pickle.load(f)
    inlier_features = [inlier_feature[opt.feature_to_use].detach().numpy() for inlier_feature in inlier_features]
    inlier_features = np.squeeze(np.array(inlier_features))

    with open(opt.outliers_feature_path, "rb") as f:
        outlier_features, outlier_labels = pickle.load(f)
    outlier_features = [outlier_feature[opt.feature_to_use].detach().numpy() for outlier_feature in outlier_features]
    outlier_features = np.squeeze(np.array(outlier_features))

    centers = []
    all_similarities_train = []
    centers = feature_stats(train_features_sorted)

    all_similarities_inliers = []
    for i in range(opt.num_classes):
        all_similarities_inliers.append(mahalanobis_distances(inlier_features, centers[i]))
    all_similarities_inliers = np.array(all_similarities_inliers)

    all_similarities_outliers = []
    for i in range(opt.num_classes):
        all_similarities_outliers.append(mahalanobis_distances(outlier_features, centers[i]))
    all_similarities_outliers = np.array(all_similarities_outliers)

    binary_scores = np.concatenate((-np.min(all_similarities_inliers, axis=0), -np.min(all_similarities_outliers, axis=0)))
    auroc_dis = AUROC(binary_labels, binary_scores)
    print("auroc_dis", auroc_dis)

