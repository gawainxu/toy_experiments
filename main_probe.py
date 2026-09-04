import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


def getArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path_train", type=str,
                        default="./features/cifar100_marco_resnet18_1trail_19_128_128_data_19_train")
    parser.add_argument("--feature_path_test", type=str, default="./features/cifar100_marco_resnet18_1trail_19_128_128_data_19_test_known")
    parser.add_argument("--remove_extra_classes", type=int, default=0)

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

    features_train = [np.squeeze(f) for f in features_head_train]
    features_test = [np.squeeze(f) for f in features_head_test]
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    labels_train = [i - min(labels_train) for i in labels_train]
    labels_test = [i - min(labels_test) for i in labels_test]

    if opt.remove_extra_classes == 1:
        classes_train = [i for i, l in enumerate(labels_train) if l < 10]
        classes_test = [i for i, l in enumerate(labels_test) if l < 10]
        features_train = features_train[classes_train]
        labels_train = [labels_train[i] for i in classes_train]
        features_test = features_test[classes_test]
        labels_test = [labels_test[i] for i in classes_test]

    accuracy = regression(features_train, labels_train, features_test, labels_test)