import pickle
from scipy.spatial.distance import mahalanobis
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import alpha


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

    num_classes = 3
    featurePath = "/home/zhi/projects/open_cross_entropy/features/E2_100"
    feature_to_visulize = "linear2"
    
    with open(featurePath, "rb") as f:
        featuresTrain, labelsTrain = pickle.load(f)
 
    featuresTrain = [featureTrain[feature_to_visulize].detach().numpy() for featureTrain in featuresTrain]
    featuresTrain = np.squeeze(np.array(featuresTrain))

    seperated_features = seperate_class(featuresTrain, labelsTrain, num_classes)
    
    centers = []
    all_similarities_train = []
    for class_features in seperated_features:
        center = feature_stats(class_features)
        centers.append(center)
        similarities = mahalanobis_distances(class_features, center)
        all_similarities_train.append(similarities)

    all_similarities_train = np.array(all_similarities_train)


    featurePath_test = "/home/zhi/projects/open_cross_entropy/features/osr_rectangle_green_1_100"
    with open(featurePath_test, "rb") as f:
        featuresTest, labelsTest = pickle.load(f)

    featuresTest = [featureTest[feature_to_visulize].detach().numpy() for featureTest in featuresTest]
    featuresTest = np.squeeze(np.array(featuresTest))
    #seperated_features_test = seperate_class(featuresTest, labelsTest, num_classes+3)

    all_similarities_test = []
    for i in range(num_classes):
       all_similarities_test.append(mahalanobis_distances(featuresTest, centers[i]))

    all_similarities_test = np.array(all_similarities_test)

    closest_class = np.argmin(np.mean(all_similarities_test, axis=1))
    print("closest_class", closest_class)

    all_similarities_train_closest = all_similarities_train[closest_class]
    all_similarities_test_closest = all_similarities_test[closest_class]

    # compute the histogram distances
    bins = np.linspace(0, 50, 200)
    hist_train, e1 = np.histogram(all_similarities_train_closest, bins)
    hist_test, e2 = np.histogram(all_similarities_test_closest, bins)

    hist_dis = np.sum(np.abs(hist_train - hist_test)) * 1.0 / len(hist_train)
    print(hist_dis)

    _, axs = plt.subplots(nrows=1, ncols=1)
    axs.hist(all_similarities_train_closest, bins=bins, label="train", alpha=0.3)
    axs.hist(all_similarities_test_closest, bins=bins, label="test", alpha=0.3)
    plt.legend()
    plt.show()

