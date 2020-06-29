import numpy as np

import torch

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.optim import lr_scheduler

import os 
import time

import models


import matplotlib.pyplot as plt

#Dataset
from dataset.cifar_dataset import CIFAR10, CIFAR100
from dataset.DataLoader import DataLoader

from torchvision import datasets, transforms
from torchvision import utils

#Attacks
from advertorch.attacks.iterative_projected_gradient import LinfPGDAttack
from advertorch.attacks.carlini_wagner import CarliniWagnerL2Attack


from attacks.iterative_projected_gradient import LinfPGDAttack as LINFPGD

from attacks.carlini_eot_logit import CarliniAttackEotLogit
from attacks.carlini_eot_loss import CarliniAttackEotLoss
from attacks.carlini_eot_probit import CarliniAttackEotProbit


class Mixture_of_Classifier(nn.Module):
    '''Mixture_of_classifier

    Attributes
    ----------

    train_loader : torch.utils.data.DataLoader
        The training dataloader
    test_loader : torch.utils.data.DataLoader
        The testing dataloader
    device : torch.device
        variable that tells if we use GPUs ('cuda') or CPUs ('cpu')
    config : dict
        object that contains all the hyperparameters we need (has to be as in the ./config.json file)

    Methods
    -------
    load(...)
        Load the classifiers available

    boosting(...)
        Method that apply the BAT algorithm

    addBoostedClassifier(...)
        Method called in boosting(...) to create the new classifier

    trainBoostedClassifier(...)
        Method called in boosting(...) to train the new classifier

    adversarialTrainLoader(...)
        Method called in trainBoostedClassifier(...) to create the Adversarial Data set against the previous mixture

    updateWeights(...)
        Method that update the weights of the classifier with the parameters config['alpha']

    test_mixture_accuracy_under_attack(...)
        test the accuracy under PGD or C&W attack of the mixture

    test_mixture_accuracy_under_carlini_attack(...)
        test the accuracy under CW attack (by doing max of different EOT)

    test_classifier_accuracy_under_attack(...)
        test the accuracy under PGD or C&W attack of a choosen classifier

    test_mixture_accuracy(...)
        test the natural accuracy of the mixture

    predict(...)
        gives a batch of example to a choosen classifier and return its outputs

    forward(...)
        give a batch of sample to the mixture and return its ouput

    '''
    def __init__(self, train_loader=None, test_loader=None, device=None, config = None):
        super(Mixture_of_Classifier, self).__init__()

        print("Creation of the Mixture class")

        self.train_loader = train_loader 
        self.test_loader = test_loader
        self.device = device
        self.config = config


        self.classifiers = []
        self.number_of_models = self.config['number_of_models']
        self.save_dir = self.config['save_dir'] + "/" + self.config['dataset']
        
        if not os.path.isdir(self.config['save_dir']) : 
            os.mkdir(self.config['save_dir'])
        if not os.path.isdir(self.config['save_dir'] + "/" + self.config['dataset']) : 
            os.mkdir(self.config['save_dir'] + "/" + self.config['dataset'])
        if not os.path.isdir(self.save_dir) :
            os.mkdir(self.save_dir)
            
        self.weights = np.ones(0)

    def load(self, top_accuracy_under_attack = False, level = 0):
        print(level*"   "+"Loading models...")

        if top_accuracy_under_attack == True :
            load_dir = self.save_dir+"/topAccuracyUnderAttack"
        else :
            load_dir = self.save_dir+"/topAccuracy"
        
        for i in range(self.number_of_models):
            if not os.path.exists(load_dir+"/model_"+str(i)+".pth"):
                self.resume_epoch = -1
                return False
           
            classifier, accuracy, accuracy_under_attack = models.load_model(self.save_dir, i, -1, self.device, self.config['number_of_class'], top_acc_under_attack=top_accuracy_under_attack)
            print((level+1)*"   "+"Model n°"+str(i)+" loaded. Accuracy : %.2f - Accuracy Under Attack : %.2f " % (accuracy, accuracy_under_attack))
            self.classifiers.append(classifier)
            
            
                                    
            self.updateWeights(alpha = self.config['alpha'])

            print((level+1)*"   "+"Weights of the current mixture : "+str(self.weights))
            

            topAccuracy = -1
            topAccuracy_bis = -1
            topAccuracyUnderAttack=-1
            toAccuracyUnderAttack_bis = -1

            if not os.path.exists(self.save_dir+"/model_"+str(i)+"/epoch_"+str(self.config['epochs']-1)+".pth"):
                print((level+1)*"   "+"Classifier not completely trained")
                for j in range(200):
                    if not os.path.exists(self.save_dir+"/model_"+str(i)+"/epoch_"+str(j)+".pth"):
                        classifier, accuracy, accuracy_under_attack = models.load_model(self.save_dir, i, j-1, self.device, self.config['number_of_class'], top_acc_under_attack=top_accuracy_under_attack)

                        classifier.module.best_acc = topAccuracy
                        classifier.module.best_accuracy_under_attack = topAccuracyUnderAttack

                        self.classifiers[-1] = classifier
                        self.resume_epoch = j
                        
                        return False
                    else :
                        classifier, accuracy, accuracy_under_attack = models.load_model(self.save_dir, i, j, self.device, self.config['number_of_class'], top_acc_under_attack=top_accuracy_under_attack)

                        if topAccuracy < accuracy:
                            topAccuracy = accuracy
                            topAccuracy_bis = accuracy_under_attack

                        if topAccuracyUnderAttack < accuracy_under_attack:
                            topAccuracyUnderAttack = accuracy_under_attack
                            topAccuracyUnderAttack_bis = accuracy

            else : 
                self.resume_epoch = -1

        return True

    #BAT algorithm
    def boosting(self, level=0):
        print(level*"   "+"Starting BAT algorithm...")
        
        if self.resume_epoch != -1 :
            start = len(self.classifiers) - 1
        else : 
            start = len(self.classifiers)

        for i in range(start, self.number_of_models):
            self.train_loader.shuffle = True
            if self.resume_epoch == -1:
                classifier = self.addBoostedClassifier(level = level+1)
            else : 
                classifier = self.classifiers[-1]

            self.trainBoostedClassifier(classifier = classifier, level = level +1)
            
            self.updateWeights(alpha = self.config['alpha'])
            
            self.resume_epoch = -1

    def addBoostedClassifier(self, level=0):
        print(level*"   "+"Adding Boosted Classifier n°"+str(len(self.classifiers))+"...")

        newClassifier = models.Wide_ResNet(28, 10, 0.3, self.config['number_of_class']).to(self.device)

        if self.device == torch.device('cuda'):
            newClassifier = torch.nn.DataParallel(newClassifier, range(torch.cuda.device_count()))

        return newClassifier

    def trainBoostedClassifier(self, classifier, level = 0):

        
        milestones = self.config['milestones']
        lr = self.config['lr']
        
        if self.resume_epoch != -1 :
 
            self.classifiers = self.classifiers[:-1]
            self.weights = self.weights[:-1]

            start = self.resume_epoch

            tmp = -1
            for m in range(len(milestones)):
                if milestones[m] <= self.resume_epoch :
                    lr = lr * self.config['gamma']
                    tmp = m
                else :
                    break

            if tmp != -1 :
                milestones = milestones[tmp:]

            milestones = list(np.array(milestones) - self.resume_epoch)
        else : 
            start = 0
        
        id_classifier = len(self.classifiers)

        print(level*"   "+"Training Boosted Classifier n°"+str(id_classifier)+"...")

        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=self.config['gamma'], last_epoch=-1)
     
        #Adversarial training for the first classifier
        if  id_classifier == 0 :
            
            attack = LinfPGDAttack( classifier, 
                                    eps=self.config['eps']/255, 
                                    eps_iter=self.config['eps_iter']/255, 
                                    nb_iter=self.config['nb_iter'], 
                                    rand_init=self.config['rand_init'], 
                                    clip_min=self.config['clip_min'], 
                                    clip_max=self.config['clip_max'])
            
            for epoch in range(start, self.config['epochs']):
                classifier.train()

                models.adversarialTrain(classifier, self.device, self.train_loader, optimizer, epoch, attack, level = level + 1)
                
                scheduler.step()
                
                classifier.eval()
                
                accuracy_under_attack = models.test_under_attack(classifier, self.device, self.test_loader, attack, level = level +1)
                accuracy = models.test(classifier, self.device, self.test_loader, level = level +1)
                
                models.save_model(self.save_dir, id_classifier, self.device, classifier, accuracy, accuracy_under_attack, epoch, level = level +1)

                models.updateAndSaveBestAccuracies(self.save_dir, id_classifier, self.device, classifier, accuracy, accuracy_under_attack, level = level +1)
                
        else : #Natural training on the adversarial data set created against the mixture

            adversarial_train_loader, adversarial_test_loader = self.adversarialTrainLoader(level = level + 1)

            for epoch in range(start, self.config['epochs']):
                classifier.train()

                models.train(classifier, self.device, adversarial_train_loader, optimizer, epoch, level = level + 1)
            
                scheduler.step()
                
                classifier.eval()

                accuracy_under_attack = models.test(classifier, self.device, adversarial_test_loader, level = level +1)
                accuracy = models.test(classifier, self.device, self.test_loader, level = level +1)
                
                models.save_model(self.save_dir, id_classifier, self.device, classifier, accuracy, accuracy_under_attack, epoch, level = level +1)

                models.updateAndSaveBestAccuracies(self.save_dir, id_classifier, self.device, classifier, accuracy, accuracy_under_attack, level = level +1)


        classifier, acc, top_acc_under_attack = models.load_model(self.save_dir, id_classifier, -1, self.device, self.config['number_of_class'], top_acc_under_attack=True)

        self.classifiers.append(classifier)    
    
    def adversarialTrainLoader(self, level = 0):
        print(level*"   "+"Creating the adversarial data loader (according to the current mixture)...")
        attack = LINFPGD(   self, 
                            eps=self.config['eps']/255, 
                            eps_iter=self.config['eps_iter']/255, 
                            nb_iter=self.config['nb_iter'], 
                            rand_init=self.config['rand_init'], 
                            clip_min=self.config['clip_min'], 
                            clip_max=self.config['clip_max'])

        #Training Data set
        if self.config['dataset'] == "cifar10":
            data_class = CIFAR10
        elif self.config['dataset'] == "cifar100":
            data_class = CIFAR100

        train_loader = DataLoader(
            data_class(
                root=self.config['dataroot'], 
                train=True,
                download=True, 
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])
            ), 
            batch_size=self.train_loader.batch_size, 
            shuffle=True, 
            num_workers= 2, 
            drop_last= True
        )

        batch_size = self.train_loader.batch_size


        for i in range(len(train_loader)):
            
            train_loader.dataset.data[np.arange(i*batch_size, (i+1)*batch_size)] = attack.perturb(train_loader.dataset.data[np.arange(i*batch_size, (i+1)*batch_size)].cuda()).cpu()
        #Testing Dataset
        test_loader = DataLoader(
            data_class(
                root=self.config['dataroot'], 
                train=False,
                download=True, 
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])
            ), 
            batch_size=self.test_loader.batch_size, 
            shuffle=True, 
            num_workers= 2, 
            drop_last= True
        )

        batch_size = self.test_loader.batch_size
        for i in range(len(test_loader)):
         
            test_loader.dataset.data[np.arange(i*batch_size, (i+1)*batch_size)] = attack.perturb(test_loader.dataset.data[np.arange(i*batch_size, (i+1)*batch_size)].cuda()).cpu()
        return train_loader, test_loader

    def updateWeights(self, alpha=0.1):
        if len(self.weights)==0:
            self.weights = np.ones(1)
        else : 
            tmp = self.weights
            self.weights = np.ones(len(self.classifiers))
            self.weights[np.arange(len(tmp))] = tmp
            
            self.weights[np.arange(len(self.classifiers)-1)] *= (1-alpha)
            self.weights[len(self.classifiers)-1] = alpha    
    def test_mixture_accuracy_under_carlini_attack(self, level = 0):

        attacks = []


        attacks.append( CarliniAttackEotLogit(   self,
                                            num_classes = self.config['number_of_class'],
                                            learning_rate = self.config['learning_rate'], 
                                            binary_search_steps = self.config['binary_search_steps'],
                                            max_iterations = self.config['max_iterations'], 
                                            abort_early = self.config['abort_early'],
                                            initial_const = self.config['initial_const'], 
                                            clip_min = self.config['clip_min'], 
                                            clip_max = self.config['clip_max']))
        
        attacks.append(CarliniAttackEotProbit(  self,
                                            num_classes = self.config['number_of_class'],
                                            learning_rate = self.config['learning_rate'], 
                                            binary_search_steps = self.config['binary_search_steps'],
                                            max_iterations = self.config['max_iterations'], 
                                            abort_early = self.config['abort_early'],
                                            initial_const = self.config['initial_const'], 
                                            clip_min = self.config['clip_min'], 
                                            clip_max = self.config['clip_max']))


        attacks.append(CarliniAttackEotLoss(self,
                                            num_classes = self.config['number_of_class'],
                                            learning_rate = self.config['learning_rate'], 
                                            binary_search_steps = self.config['binary_search_steps'],
                                            max_iterations = self.config['max_iterations'], 
                                            abort_early = self.config['abort_early'],
                                            initial_const = self.config['initial_const'], 
                                            clip_min = self.config['clip_min'], 
                                            clip_max = self.config['clip_max']))


        accuracy = models.test_under_carlini_attack(self, self.device, self.test_loader, attacks, level = level+1)

        return accuracy

   
    def test_mixture_accuracy_under_attack(self, adversary = "pgd", number_of_models = None, level = 0):

        if number_of_models == None : 
            number_of_models = self.number_of_models
       
        tmp_classifiers = self.classifiers
        self.classifiers = self.classifiers[:number_of_models]

        if adversary == "pgd" :
            mixed_attack = LINFPGD( self, 
                                    eps=self.config['eps']/255, 
                                    eps_iter=self.config['eps_iter']/255, 
                                    nb_iter=self.config['nb_iter'], 
                                    rand_init=self.config['rand_init'], 
                                    clip_min=self.config['clip_min'], 
                                    clip_max=self.config['clip_max'])
        elif adversary == "carlini": 
            mixed_attack = CarliniAttackEotLoss(self,
                                            num_classes = self.config['number_of_class'],
                                            learning_rate = self.config['learning_rate'], 
                                            binary_search_steps = self.config['binary_search_steps'],
                                            max_iterations = self.config['max_iterations'], 
                                            abort_early = self.config['abort_early'],
                                            initial_const = self.config['initial_const'], 
                                            clip_min = self.config['clip_min'], 
                                            clip_max = self.config['clip_max'])
        
        accuracy = models.test_under_attack(self, self.device, self.test_loader, mixed_attack,adversary = adversary, level = level +1)
        self.classifiers = tmp_classifiers
        
        return accuracy

    def test_classifier_accuracy_under_attack(self, i, adversary = "pgd", level = 0):

        if adversary == "pgd" :
            attack = LinfPGDAttack( self.classifiers[i], 
                                    eps=self.config['eps']/255, 
                                    eps_iter=self.config['eps_iter']/255, 
                                    nb_iter=self.config['nb_iter'], 
                                    rand_init=self.config['rand_init'], 
                                    clip_min=self.config['clip_min'], 
                                    clip_max=self.config['clip_max'])
        elif adversary == "carlini": 
            attack = CarliniWagnerL2Attack( self.classifiers[i],
                                            num_classes = self.config['number_of_class'],
                                            learning_rate = self.config['learning_rate'], 
                                            binary_search_steps = self.config['binary_search_steps'],
                                            max_iterations = self.config['max_iterations'], 
                                            abort_early = self.config['abort_early'],
                                            initial_const = self.config['initial_const'], 
                                            clip_min = self.config['clip_min'], 
                                            clip_max = self.config['clip_max'])

        accuracy = models.test_under_attack(self.classifiers[i], self.device, self.test_loader, attack, adversary = adversary, level = level +1)
        
        return accuracy

    def test_mixture_accuracy(self, level = 0):
        accuracy = models.test(self, self.device, self.test_loader, level = self.level +1)
        
        return accuracy

    def predict(self, x, choosen_classfier_id):

        return self.classifiers[choosen_classfier_id](x)

    def forward(self, x, with_choosen_classifier_id = False): 

        batch_of_logit = torch.zeros(len(x), self.config['number_of_class']).cuda()

        p = self.weights/np.sum(self.weights)

        choosen_ids = np.random.choice(len(self.classifiers), len(x), p=p)


        for i in range(len(self.classifiers)):
            indices = (choosen_ids==i).nonzero()[0]
            batch_of_logit[indices] = self.predict(x, i)[indices]

        if with_choosen_classifier_id == False :
            return batch_of_logit
        else : 
            return batch_of_logit, choosen_ids


