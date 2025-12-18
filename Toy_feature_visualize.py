#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:57:45 2020

@author: zhi
"""


import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle 

import torch
import torch.nn as nn

from feature_analysis import sort_features, sort_features_class
from Toy_model import toy_model, toy_model_supcon


def pca(inMat, nComponents):
    
    # It is better to make PCA transformation before tSNE
    pcaFunction = PCA(nComponents)
    outMat = pcaFunction.fit_transform(inMat)

    return outMat    
    
    

def tSNE(inMat, nComponents):
    """
    The function used to visualize the high-dimensional hyper points 
    with t-SNE (t-distributed stochastic neighbor embedding)
    https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0
    https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    """
    
    inEmbedded = TSNE(n_components=nComponents, perplexity=10).fit_transform(inMat)
    return inEmbedded
    
    
    
if __name__ == "__main__":
    
    
    
    inlier_features_path = "/home/zhi/projects/open_cross_entropy/features/toy_features_inliers_train_supcon_E2_399"  #
    outlier_features_path = "/home/zhi/projects/open_cross_entropy/features/toy_features_outliers_supcon_E2_399"      #

    feature_to_visulize = "linear2"
    num_classes = 2               #
    loss = 'supcon'               #
    num_dim = 20
    
    if loss == 'CE': 
        model = toy_model(num_classes)
    else:
        model = toy_model_supcon(num_dim)
        
    model_path = "../save/toy_model_supcon_E2_399"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with open(inlier_features_path, "rb") as f:
        inlier_features, inlier_labels = pickle.load(f)

    with open(outlier_features_path, "rb") as f:
        outlier_features, outlier_labels = pickle.load(f)

    #print(len(inlier_features))

    sorted_inlier_features_in_dict = sort_features(inlier_features)
    sorted_outlier_features_in_dict = sort_features(outlier_features)

    features_to_view_inliers = sorted_inlier_features_in_dict[feature_to_visulize]
    features_to_view_outliers = sorted_outlier_features_in_dict[feature_to_visulize]
    
    if "conv" in feature_to_visulize:
       features_to_view_inliers = np.reshape(features_to_view_inliers, (features_to_view_inliers.shape[0], -1))

        
    labelsTest = inlier_labels + outlier_labels
    allFeatures = features_to_view_inliers + features_to_view_outliers
    
    featuresSNE = np.squeeze(np.array(allFeatures))                                     
    #print(featuresSNE.shape)
    #featuresSNE = pca(featuresTest, 10)       #10
    featuresSNE = tSNE(featuresSNE, 2)
    #featuresSNE = np.concatenate((featuresSNE, features), 0)

    #featuresSNE = np.squeeze(featuresTest)
    
    allLabels = []
    for l in labelsTest:
        if l >= num_classes:
            allLabels.append(1000)
        else:
            allLabels.append(l)
    
    allLabels = np.squeeze(np.reshape(np.array(allLabels), [1, -1]))
    print(allLabels)
    
    f = {"feature_1": featuresSNE[:, 0], 
         "feature_2": featuresSNE[:, 1],
         "label": allLabels}
    
    fp = pd.DataFrame(f)
    
    a4_dims = (8, 6)
    fig, ax = plt.subplots(figsize=a4_dims)
    
    sns.scatterplot(ax=ax, x="feature_1", y="feature_2", hue="label",
                    palette=['blue','red', "k"], data=fp, 
                    legend="full", alpha=0.5)
    fig.savefig("../plots/inliers_train_supcon_E2_399.png")
    
    
    """
    https://medium.com/swlh/how-to-create-a-seaborn-palette-that-highlights-maximum-value-f614aecd706b
    
    'green','orange','brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey'
    ,'brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey',
                            'rosybrown', 'm', 'y', 'tan', 'lime', 'azure', 'sky', 'darkgreen',
                            'grape', 'jade'
    sns.color_palette("hls", num_classes)
    """