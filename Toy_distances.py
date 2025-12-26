import pickle
import sys

from scipy.spatial.distance import mahalanobis
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_options():
    parser = argparse.ArgumentParser("Arguments")

    parser.add_argument("--feature_path", type=str, default="./features/toy_toy_E1_0_train")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--feature_path_test", type=str, default="./features/toy_toy_E1_0_rectangle_blue")
    parser.add_argument("--feature_to_visualize", type=str, default="linear2")
    parser.add_argument("--fig_save_path", type=str, default="./plots/hist_E1_0.png")

    opt = parser.parse_args()
    return opt


def feature_stats(features):

    features = np.squeeze(np.array(features))
    mu = np.mean(features, axis=0)
    var = np.cov(features.astype(float), rowvar=False)

    return mu, var

def cosine_distance(features, center):

    """
    features: [num_samples, feature_dim]
    center: [feature_dim]
    """
    features = np.array(features)
    centers = np.tile(center, (len(features), 1))
    similarities = np.matmul(features, centers.T)[:, 0]
    center_norm = np.linalg.norm(center)
    similarities = similarities / center_norm
    features_norm = np.linalg.norm(features, axis=1)
    similarities = np.divide(similarities, features_norm)

    return similarities


def mahalanobis_distances(features, stats):

    features = np.array(features)
    mu, var = stats
    diss = []
    for feature in features:
        features_normalized = feature - mu
        dis = mahalanobis(feature, mu, np.linalg.inv(var))
        diss.append(dis)

    return  diss


def seperate_class(featuresTest, labelsTest, num_classes):

    seperated_features = [[] for i in range(num_classes)]
    for i, l in enumerate(labelsTest):
        #print(featuresTest[i])
        seperated_features[l].append(featuresTest[i])

    return seperated_features


if __name__ == "__main__":

    opt = parse_options()
    
    with open(opt.feature_path, "rb") as f:
        featuresTrain, labelsTrain = pickle.load(f)
 
    featuresTrain = [featureTrain[opt.feature_to_visualize].detach().numpy() for featureTrain in featuresTrain]
    featuresTrain = np.squeeze(np.array(featuresTrain))

    seperated_features = seperate_class(featuresTrain, labelsTrain, opt.num_classes)
    
    centers = []
    all_similarities_train = []
    for class_features in seperated_features:
        center = feature_stats(class_features)
        centers.append(center)
        similarities = mahalanobis_distances(class_features, center)
        all_similarities_train.append(similarities)

    all_similarities_train = np.array(all_similarities_train)

    with open(opt.feature_path_test, "rb") as f:
        featuresTest, labelsTest = pickle.load(f)

    featuresTest = [featureTest[opt.feature_to_visualize].detach().numpy() for featureTest in featuresTest]
    featuresTest = np.squeeze(np.array(featuresTest))
    #seperated_features_test = seperate_class(featuresTest, labelsTest, num_classes+3)

    all_similarities_test = []
    for i in range(opt.num_classes):
       all_similarities_test.append(mahalanobis_distances(featuresTest, centers[i]))

    all_similarities_test = np.array(all_similarities_test)

    closest_class = np.argmin(np.mean(all_similarities_test, axis=1))
    print("closest_class", closest_class)

    all_similarities_train_closest = all_similarities_train[closest_class]
    all_similarities_test_closest = all_similarities_test[closest_class]

    # compute the histogram distances
    bins = np.linspace(0, 10, 200)
    hist_train, e1 = np.histogram(all_similarities_train_closest, bins)
    hist_test, e2 = np.histogram(all_similarities_test_closest, bins)

    hist_dis = np.sum(np.abs(hist_train - hist_test)) * 1.0 / len(hist_train)
    print(hist_dis)

    _, axs = plt.subplots(nrows=1, ncols=1)
    axs.hist(all_similarities_train_closest, bins=bins, label="train", alpha=0.3)
    axs.hist(all_similarities_test_closest, bins=bins, label="test", alpha=0.3)
    plt.legend()
    plt.savefig(opt.fig_save_path)
    sys.exit(hist_dis)

