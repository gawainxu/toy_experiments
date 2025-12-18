# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--img_size', type=int, default=32,
                        help='size of the input image')
    parser.add_argument('--model', type=str, default="vgg",
                        help='Model name')
    parser.add_argument('--task', type=str, default="[6,7]",               
                        required=False, help='task list')
    parser.add_argument('--num_total_classes', type=int, default=10,
                        help='total number of classes')
    parser.add_argument('--lr', type=float, default=1e-2, 
                        required=False, help='Learning rate.')
    
    parser.add_argument('--if_train', type=str, default="feature", 
                        help='if training or testing or reading features')
    
    parser.add_argument('--testing_model_path', type=str, 
                        default="../save/cifar10_vgg_lwf_1.pth", 
                        help='if training or testing')
    parser.add_argument('--testing_task', type=int, default=1, 
                        help='if training or testing')
    
    
    parser.add_argument('--feature_reading_model_path', type=str, 
                        default="../save/cifar10_vgg_lwf_0.pth", 
                        help='if training or testing')
    parser.add_argument('--feature_save_path', type=str, 
                        default="../save/feature_linear_6_lwf_cifar10_class2", 
                        help='index of the layer to read feature')
    parser.add_argument('--feature_reading_task', type=int, default=0, 
                        help='if training or testing')
    

    parser.add_argument('--optim_wd', type=float, default=5e-4,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    


    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers.')
    parser.add_argument('--init_model_path', type=str, default='',
                        help='The batch size of the memory buffer.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    parser.add_argument('--exemplar_file', type=str, default="../save/exemplar",
                        help='path to save exemplars')
    
    
def add_plain_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for plain method
    """
    parser.add_argument('--replay_buff_size', type=int, default=500,
                        help='buff size in icarl')
    pass



def add_cifar10_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for cifar10 dataset
    """
    pass


def add_cifar100_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for cifar10 dataset
    """
    pass


def add_mnist_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for cifar10 dataset
    """
    pass



def add_vgg_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for vgg model
    """
    parser.add_argument('--vgg_type', type=str, default="vgg11",
                        help='which type of vgg it is')


def add_resnet_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used for cifar10 dataset
    """
    pass