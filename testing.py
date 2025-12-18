import torch
from torchvision import transforms, datasets

import numpy as np
import pickle
from datasets.utils.utils import mapping
from datasets.cifar10 import myCIFAR10
from datasets.dataset import customDataset
from torch.utils.data import DataLoader
import itertools


datasets_dict = {"cifar10": myCIFAR10}


def features_reading(model, data_loader, tested_classes):

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            print("hook working!!!")
            activation[name] = output.detach()
        return hook

    #model.feature_extractor.register_forward_hook(get_activation(feature_name))
    model.eval()

    outputs = [[] for i in range(tested_classes)]
    for i, (img, label) in enumerate(data_loader):
        
        img = img.cuda()
        outputs[label.item()].append(model.feature_extractor(img).data.cpu().numpy())

    return outputs


def exemplar_centers(features):

    center_features = []
    for c_features in features:
        center = np.mean(np.array(c_features), axis=0)
        center_features.append(center)

    return center_features


def accuracy(outputs, targets):
    
    unequ = 0
    for pred, target in zip(outputs, targets):
        
        pred = np.argmax(pred)
        if pred != target:
              print(pred)
              unequ += 1
        
    return 1 - unequ*1.0 / len(outputs)


def testing(model, args, method):
    
    tasks_to_test = args.tasks[ : args.testing_task + 1 ]   
    tested_classes = [c for t in tasks_to_test for c in t]
    print(tested_classes)
    print(tasks_to_test)
    sorted_labels, label_mapping = mapping(tasks_to_test)
    print(sorted_labels, label_mapping)
    sorted_labels = list(itertools.chain(*sorted_labels))
    print(sorted_labels)
    
    # TODO it is the case without jumping labels
    dataset = datasets_dict[args.dataset](root="../../data", classes=tested_classes,                     
                                          train=False, download=True, args=args)
    exemplar_class_centers = None

    data_loader = dataset.get_dataloader()
    model.load_state_dict(torch.load(args.testing_model_path))
    model.eval()
    model = model.cuda()
  
  
    if args.method == "icarl":
        with open(args.icarl_exemplar_path, "rb") as f:
             exemplar_sets, exemplar_labels = pickle.load(f)

        exemplar_dataset = customDataset(exemplar_sets, exemplar_labels, args.dataset, transform=None)
        exemplar_dataloader = DataLoader(exemplar_dataset, batch_size = 1, 
                                         num_workers = 4, shuffle = False)
        
        exemplar_features = features_reading(model, exemplar_dataloader, len(tested_classes))
        exemplar_class_centers = exemplar_centers(exemplar_features)
        method.exemplar_class_centers = exemplar_class_centers
    
    
    targets = []
    predictions = []
    for i, (inputs, labels) in enumerate(data_loader):
        
        outputs = method.classify(inputs, exemplar_class_centers)
        predictions.append(outputs.cpu().detach().numpy())
        targets.append(labels.detach().numpy())
        print(i, labels)        

    return accuracy(predictions, targets)
