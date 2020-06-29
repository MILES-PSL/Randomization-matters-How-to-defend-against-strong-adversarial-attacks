from __future__ import print_function
import argparse
import os
import json
import sys

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision import utils

from mixture import Mixture_of_Classifier

import time
import matplotlib.pyplot as plt

from dataset.cifar_dataset import CIFAR10, CIFAR100
from dataset.DataLoader import DataLoader


#get the config file for default values
with open('config.json') as config_file:
    config = json.load(config_file)

parser = argparse.ArgumentParser(description="test")

parser.add_argument('--adversary', default='pgd', required=True, help="choose 'pgd' or 'carlini'")
parser.add_argument('--alpha', type=float, default=config['alpha'])

args = parser.parse_args()

config['alpha']=args.alpha

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading datasets                                                                                                              #

config['dataroot'] = config['dataroot']+"/"+config['dataset']

if config['dataset']=='cifar10':
    custom_data_class=CIFAR10
    original_data_class = datasets.CIFAR10
    config['number_of_class']=10
elif config['dataset'] == 'cifar100':
    custom_data_class=CIFAR100
    original_data_class = datasets.CIFAR100
    config['number_of_class']=100

train_loader = DataLoader(
    custom_data_class(
        root=config['dataroot'], 
        train=True,
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ), 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers= 2, 
    drop_last= True
)

test_loader = torch.utils.data.DataLoader(
    original_data_class(
        root=config['dataroot'], 
        train=False, 
        transform=transforms.Compose([
                   transforms.ToTensor()
        ])),
    batch_size=config['test_batch_size'], 
    shuffle=False, 
    num_workers= 2, 
    drop_last= True
)

#Build the mixture class

MC = Mixture_of_Classifier( train_loader = train_loader, 
                            test_loader = test_loader, 
                            device = device, 
                            config = config)


#For printing
level = 0

#Boolean variable that help us to know if we have enough saved classifier to load.
completeLoading = False
if config['load'] == True : 
    completeLoading = MC.load(top_accuracy_under_attack= True, level = level + 1)

if completeLoading == False : 
    sys.exit("Not enough saved classifiers")

if config["number_of_models"] > 1 :
    if args.adversary == 'pgd':
        accuracy = MC.test_mixture_accuracy_under_attack(adversary = 'pgd', level = level + 1)
    elif args.adversary == 'carlini':
        accuracy = MC.test_mixture_accuracy_under_carlini_attack(level = level + 1)
else :
    accuracy = MC.test_classifier_accuracy_under_attack(0, adversary = args.adversary , level = level + 1)


if not os.path.isdir(MC.save_dir+"/eval"):
    os.mkdir(MC.save_dir+"/eval")

if not os.path.isdir(MC.save_dir+"/eval/"+str(config['alpha'])):
    os.mkdir(MC.save_dir+"/eval/"+str(config['alpha']))

save_dir = MC.save_dir+"/eval/"+str(config['alpha'])
#Saving the results
if args.adversary == 'pgd':
    torch.save(accuracy, save_dir + "/" + args.adversary+".pth")
elif args.adversary == 'carlini':
    for i in range(len(accuracy)):
        torch.save(accuracy[i][1], save_dir+"/"+args.adversary+"_threshold_"+str(accuracy[i][0])+".pth")
        
        if accuracy[i][0] == 0.8 :
            torch.save(accuracy[i][1], save_dir + "/" + args.adversary+".pth")



