import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


def getArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path_train", type=str,
                        default="./features/cifar100_marco_resnet18_1trail_0_128_128_data_3_train")
    parser.add_argument("--feature_path_test", type=str, default="./features/cifar100_marco_resnet18_1trail_0_128_128_data_3_test")

    opt = parser.parse_args()

    return opt


def regression(train_features, train_labels, test_features, test_labels):
    clf = LogisticRegression(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=0, )

    clf.fit(train_features, train_labels)
    accuracy = clf.score(test_features, test_labels)
    print("accuracy:", accuracy)

    return accuracy


if __name__ == "__main__":

    opt = getArgs()

    with open(opt.feature_path_train, "rb") as f:
        features_head_train, features_backbone_train, labels_train = pickle.load(f)

    with open(opt.feature_path_test, "rb") as f:
        features_head_test, features_backbone_test, labels_test = pickle.load(f)

    features_train = [np.squeeze(f.numpy()) for f in features_head_train]
    features_test = [np.squeeze(f.numpy()) for f in features_head_test]
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    labels_train = [i - min(labels_train) for i in labels_train]
    labels_test = [i - min(labels_test) for i in labels_test]

    accuracy = regression(features_train, labels_train, features_test, labels_test)