import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import init

import os
import sys

import random

import numpy as np


#The methods conv3x3, conv_init, wide_basic and the class Wide_Resnet 
#have been taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
#We have modified the class Wide_Resnet.

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.leaky_relu(self.bn1(x), negative_slope=0.1)))
        out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=0.1))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        self.best_acc = -1
        self.best_accuracy_under_attack = -1

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.leaky_relu(self.bn1(out), negative_slope=0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def updateBestAccuracies(self, accuracy, accuracy_under_attack):
        acc = False
        acc_under_attack = False

        if accuracy > self.best_acc :
            self.best_acc = accuracy
            acc = True
        if accuracy_under_attack > self.best_accuracy_under_attack :
            self.best_accuracy_under_attack = accuracy_under_attack
            acc_under_attack = True

        return acc, acc_under_attack


#Method to naturally train a classifier
def train(model, device, train_loader, optimizer, epoch, level = 0):

    criterion = nn.CrossEntropyLoss()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(level*"   "+"Epoch : "+str(epoch))

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(level*"   "+'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        



#method to adversarially train a classifier
def adversarialTrain(model, device, train_loader, optimizer, epoch, attack, level = 0):
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(level*"   "+"Epoch : "+str(epoch))

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = attack.perturb(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(level*"   "+'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        



#method to test a classifier
def test(model, device, test_loader, level = 0):

    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            total += targets.size(0)

            if hasattr(model, 'classifiers'):
                for i in range(len(model.classifiers)):
                    outputs = model.predict(inputs, i)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item() * model.weights[i]
            else : 
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
           
        
            print(level*"   "+'Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))

    
    return 100.*correct/total


def test_under_carlini_attack(model, device, test_loader, attacks, level = 0):

    model.eval()
    
    threshold = np.array([0.4, 0.6, 0.8])

    predictions = torch.zeros(len(threshold), len(attacks), len(test_loader), 2, test_loader.batch_size).cuda()

    mixture_output = torch.zeros(len(threshold), len(attacks), len(test_loader), test_loader.batch_size).cuda()
    
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        total += targets.size(0)
        
        for id_attack, attack in enumerate(attacks) :

            print("Attack n°"+str(id_attack+1))

            adv_inputs = attack.perturb(inputs.clone(), targets.clone())
          
            delta = inputs.clone() - adv_inputs.clone()
            l2_norm = torch.norm(delta.view(len(delta), -1), p = 2, dim = 1)

            for k in range(len(threshold)) :
                
                adv_inputs_tmp = adv_inputs.clone()

                compt = 0
                for i in range(len(inputs)):
                    if l2_norm[i] > threshold[k] :
                        compt = compt + 1
                        adv_inputs_tmp[i] = inputs.clone()[i]

                #print(level*"   "+"Number of adversarial example rejected  = ", compt)

                with torch.no_grad():

                    for i in range(len(model.classifiers)):
                        outputs = model.predict(adv_inputs_tmp, i)
                        _, predictions[k][id_attack][batch_idx][i] = outputs.max(1)

                    mixture_output[k][id_attack][batch_idx] = predictions[k][id_attack][batch_idx][0].eq(targets).int() * model.weights[0] + predictions[k][id_attack][batch_idx][1].eq(targets).int() * model.weights[1]

        accuracy = mixture_output[:, :,:batch_idx+1,:].min(1)[0].sum(dim=1).sum(dim=1)/total
        for att in range(len(attacks)):
            print("From the "+str(att)+"th attack, for each threshold : ", (mixture_output[:, :,:batch_idx+1,:].min(1)[1].eq(att).sum(dim=1).sum(dim=1)*100)/total)
        print(accuracy)
       

    return [(threshold[t], 100.*accuracy[t].item()) for t in range(len(threshold))]


#method to test a classifier on adversarial examples (using PGD or C&W)
def test_under_attack(model, device, test_loader, attack, adversary = 'pgd', level = 0):

    model.eval()
    
    if adversary == 'carlini':
        threshold = np.array([0.4, 0.6, 0.8])
    
        totals = np.zeros(len(threshold))
        corrects = np.zeros(len(threshold))

        results = np.zeros((len(threshold), expectation_iterations))
    elif adversary =='pgd':
        correct = 0
        total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        adv_inputs = attack.perturb(inputs.clone(), targets.clone())
        
        
        if adversary == 'carlini':
            
            delta = inputs.clone() - adv_inputs.clone()
            l2_norm = torch.norm(delta.view(len(delta), -1), p = 2, dim = 1)

            
            for k in range(len(threshold)) :
                
                adv_inputs_tmp = adv_inputs.clone()

                compt = 0
                for i in range(len(inputs)):
                    if l2_norm[i] > threshold[k] :
                        compt = compt + 1
                        adv_inputs_tmp[i] = inputs.clone()[i]

                print(level*"   "+"Number of adversarial example rejected  = ", compt)

                with torch.no_grad():
                    totals[k] += targets.size(0)
                    if hasattr(model, 'classifiers'):
                        for i in range(len(model.classifiers)):
                            outputs = model.predict(adv_inputs_tmp, i)
                            _, predicted = outputs.max(1)
                            corrects[k] += predicted.eq(targets).sum().item() * model.weights[i]
                    else : 
                        outputs = model(adv_inputs_tmp)
                        _, predicted = outputs.max(1)
                        corrects[k] += predicted.eq(targets).sum().item()

                    print(level*"   "+'Threshold : %.3f%% - Acc: %.3f%% (%.3f/%d)'% (threshold[k], 100.*corrects[k]/totals[k], corrects[k], totals[k]))
                    

        elif adversary == 'pgd' :
            with torch.no_grad():
                total += targets.size(0)
                if hasattr(model, 'classifiers'):
                    for i in range(len(model.classifiers)):
                        outputs = model.predict(adv_inputs, i)
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item() * model.weights[i]
                else : 
                    outputs = model(adv_inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                
                print(level*"   "+'Acc: %.3f%% (%.3f/%d)'% (100.*correct/total, correct, total))
        


    
    if adversary == 'carlini':
            
        return [(threshold[t], 100.*corrects[t]/totals[t]) for t in range(len(threshold))]

    elif adversary == 'pgd':

        return (100.*correct/total)

#Method to save a classifier with its ntaural accuracy and accuracy under attack
def save_model(save_dir, i, device, model, accuracy, accuracy_under_attack, epoch, level = 0):   
    print(level*"   "+'Saving..')

    if device == torch.device('cuda'):
        
        checkpoint = {
            'accuracy' : accuracy,
            'accuracy_under_attack' : accuracy_under_attack,
            'model' : model.module.state_dict()
        }
    else : 
        checkpoint = {
            'accuracy' : accuracy,
            'accuracy_under_attack' : accuracy_under_attack,
            'model' : model.state_dict()
        }

    if epoch != -1 :
        if not os.path.isdir(save_dir+"/model_"+str(i)):
            os.mkdir(save_dir+"/model_"+str(i))
        torch.save(checkpoint, save_dir+"/model_"+str(i)+'/epoch_'+str(epoch)+'.pth')
    else : 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(checkpoint, save_dir+"/model_"+str(i)+'.pth')

#Method to save the weights at a given time i of the algorithm
def save_weights(save_dir, weights, i):
    checkpoint = {
        'weights' : weights        
    }
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(checkpoint, save_dir + "/weights_"+str(i)+'.pth')
    
#Method to load a classifier at a given training epoch or the best one (according to the natural accuracy or the accuracy under attack)
def load_model(save_dir, i, epoch, device, number_of_class, top_acc_under_attack=True):

    if epoch != -1 :
        checkpoint = torch.load(save_dir+"/model_"+str(i)+"/epoch_"+str(epoch)+".pth")
    elif top_acc_under_attack == False :
        checkpoint = torch.load(save_dir+"/topAccuracy/model_"+str(i)+".pth")
    else : 
        checkpoint = torch.load(save_dir+"/topAccuracyUnderAttack/model_"+str(i)+".pth")

    classifier = Wide_ResNet(28, 10, 0.3, number_of_class).to(device)

    classifier.load_state_dict(checkpoint['model'])

    if device == torch.device('cuda'):
        classifier = torch.nn.DataParallel(classifier, range(torch.cuda.device_count()))

    classifier.eval()

    return classifier, checkpoint['accuracy'], checkpoint['accuracy_under_attack']

#Method to load the weights at a given time i of the algorithm
def load_weights(save_dir, i):
    checkpoint = torch.load(save_dir+'/weights_'+str(i)+'.pth')
    return checkpoint['weights']

#Method to save a classifier if its narual accuracy is improved or if its accuracy under attack is improved
def updateAndSaveBestAccuracies(save_dir, i, device, model, accuracy, accuracy_under_attack, level = 0):

    acc, acc_under_attack = model.module.updateBestAccuracies(accuracy, accuracy_under_attack)

    if not os.path.isdir(save_dir + '/topAccuracy'):
        os.mkdir(save_dir + '/topAccuracy')
    if not os.path.isdir(save_dir + '/topAccuracyUnderAttack'):
        os.mkdir(save_dir + '/topAccuracyUnderAttack')

    if acc == True :
        save_model(save_dir + '/topAccuracy', i, device, model, accuracy, accuracy_under_attack, -1, level = level) 

    if acc_under_attack == True : 
        save_model(save_dir + '/topAccuracyUnderAttack', i, device, model, accuracy, accuracy_under_attack, -1, level = level) 
