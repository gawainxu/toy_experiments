import pickle

import numpy as np



def sort_features(num_classes, features, labels):
    sorted_features = []

    for i in range(num_classes):
        features_i = []
        for f, l in zip(features, labels):
            if l == i:
                features_i.append(f)
        sorted_features.append(features_i)

    sorted_features = np.array(sorted_features)
    return sorted_features


def hsic(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    n = matrix_x.shape[0]
    matrix_h = np.identity(n) - (1.0 / n) * np.ones((n, n))

    x_times_h = np.matmul(matrix_x, matrix_h)
    y_times_h = np.matmul(matrix_y, matrix_h)

    return 1.0 / ((n - 1) ** 2) * np.trace(np.matmul(x_times_h, y_times_h))


def linear_cka(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    if matrix_x.ndim > 2:
        matrix_x = matrix_x.reshape([matrix_x.shape[0], -1])

    if matrix_y.ndim > 2:
        matrix_y = matrix_y.reshape([matrix_y.shape[0], -1])

    # First center the columns
    matrix_x = matrix_x - np.mean(matrix_x, 0)
    matrix_y = matrix_y - np.mean(matrix_y, 0)

    matrix_x = np.matmul(matrix_x, matrix_x.T)
    matrix_y = np.matmul(matrix_y, matrix_y.T)

    matrix_h = hsic(matrix_x=matrix_x, matrix_y=matrix_y)
    matrix_x = np.sqrt(hsic(matrix_x=matrix_x, matrix_y=matrix_x))
    matrix_y = np.sqrt(hsic(matrix_x=matrix_y, matrix_y=matrix_y))
    return matrix_h / (matrix_x * matrix_y)


def cka_identity(features1, features2, class_idx):
    cka = []

    for i in class_idx:
        # print(i)
        cka_i = linear_cka(features1[i], features2[i])
        cka.append(cka_i)

    print("mean cka", sum(cka) / len(cka))


def load_features(features_path, feature_name):

    with open(features_path, "rb") as f:
        features, labels = pickle.load(f)

    features_to_analysis = []
    for fea in features:
        features_to_analysis.append(fea[feature_name].numpy())

    features_to_analysis = np.squeeze(np.array(features_to_analysis))

    return features_to_analysis, labels


if __name__ == "__main__":

    feature_path1 = "/home/zhi/projects/open_cross_entropy/features/E2"
    feature_path2 = "/home/zhi/projects/open_cross_entropy/features/E4"
    num_classes1 = 3
    num_classes2 = 5

    feature_name = "linear2"
    class_idx = [0, 1, 2]

    features_to_analysis1, labels1 = load_features(feature_path1, feature_name)
    features_to_analysis2, labels2 = load_features(feature_path2, feature_name)

    features_to_analysis1 = sort_features(num_classes1, features_to_analysis1, labels1)
    features_to_analysis2 = sort_features(num_classes2, features_to_analysis2, labels2)

    cka_identity(features_to_analysis1, features_to_analysis2, class_idx)





