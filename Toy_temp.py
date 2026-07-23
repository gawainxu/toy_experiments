import argparse
import pickle
from sklearn.linear_model import LogisticRegression


def getArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path_train", type=str, default="/features/cifar10_resnet18_trail_0_128_0.005_train")
    parser.add_argument("--feature_path_test", type=str, default="/features/cifar10_resnet18_trail_0_128_0.005_train")

    opt = parser.parse_args()
    return opt



def regression(train_features, train_labels, test_features, test_labels):

    clf = LogisticRegression(
    C=1.0,
    max_iter=5000,
    class_weight="balanced",
    random_state=0,)

    clf.fit(train_features, train_labels)
    accuracy = clf.score(test_features, test_labels)
    print("accuracy:", accuracy)

    return accuracy


if __name__ == "__main__":
    opt = getArgs()

    with open(opt.feature_path_train, "rb") as f:
        features_train, labels_train = pickle.load(f)

    with open(opt.feature_path_test, "rb") as f:
        features_test, labels_test = pickle.load(f)

    accuracy = regression(features_train, labels_train, features_test, labels_test)