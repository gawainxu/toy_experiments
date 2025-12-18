import torch

import pickle
import random
import numpy as np
from PIL import Image
from torch.autograd import Variable 

from datasets.dataset import get_image_class


def centerComputing(features):
    
    return np.mean(features, 0)


def read_features(images, model, transform):
    
    model = model.eval()
    #model.cuda()
    features = []
    f = []
    for img in images:
        #print(img)
        x = Variable(transform(Image.fromarray(np.uint8(img))), volatile=True)
        x = x.cuda()
        feature = model.feature_extractor(x.unsqueeze(0))
        feature = feature.cpu().data.numpy()
        f.append(feature)
        feature = feature / np.linalg.norm(feature) # Normalize
        features.append(feature[0])
        
    features = np.array(features)
    return features
    


def classExemplars_euclidean(m, images, model, transform):
 
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    distances = np.linalg.norm((features-centers), axis=1)

    ind = np.argsort(distances)[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center
    

def classExemplars_similar(m, images, model, transform):
    
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    similarities = np.matmul(features, centers.T)[:, 0]
    
    ind = np.argsort(np.abs(similarities))[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center


def classExemplars_random(m, images, model=None, transform=None):
    
    #features = read_features(images, model, transform)
    #center = centerComputing(features)
    
    ind = random.sample(range(len(images)), m)
    exemplar_set = np.array(images)[ind]
    #exemplar_features = features[ind]
    
    return exemplar_set    #, exemplar_features


def createExemplars(args, num_exemplars_per_class, num_classes_in_task, total_num_classes, original_dataset, model_old=None, transform=None):
    
    exemplar_sets = []
    exemplar_labels = [] 
    exemplar_features_sets = []
    exemplar_centers = []
        
    for c in range(0, num_classes_in_task):
        print("Class: ", c)
        if total_num_classes == 2:                                      # TODO
            c_dataset = original_dataset.get_image_class(c)
        else:
            c_dataset = get_image_class(original_dataset, c)              #original_dataset.get_image_class(c)
        exemplar_set = classExemplars_random(int(num_exemplars_per_class), c_dataset, model_old, transform)           #  classExemplars_similar(int(opt.memory_per_class), c_dataset, model_old, transform)   #
        #exemplar_center = centerComputing(exemplar_features)
        #exemplar_centers.append(exemplar_center)
        exemplar_sets.append(exemplar_set)
        #exemplar_features_sets.append(exemplar_features)
        exemplar_labels = exemplar_labels + [c]*int(num_exemplars_per_class)
        
    #exemplar_sets = np.reshape(np.array(exemplar_sets), (opt.memory_per_class*opt.num_init_classes, 1, opt.img_size, opt.img_size))          #### 1
    exemplar_sets = np.reshape(np.array(exemplar_sets), (num_exemplars_per_class*num_classes_in_task, args.img_size, args.img_size, 3)) 
    exemplar_labels = np.squeeze(np.array(exemplar_labels))   
    
    exemplar_file = args.exemplar_file + args.dataset + '_' +  args.method + '_' + str(total_num_classes)
    with open(exemplar_file, "wb") as f:
        pickle.dump((exemplar_sets, exemplar_labels), f)                               # exemplar_features_sets, exemplar_centers
    
    return exemplar_sets, exemplar_labels                                              # , exemplar_centers