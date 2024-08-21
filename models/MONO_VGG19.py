#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Core Libraries
import os
import time
import glob
import math
import csv
import copy
import pathlib
from collections import Counter
from itertools import chain

# Data Manipulation
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from pylab import savefig, rcParams

# PyTorch Libraries
import torch
from torch import nn, flatten
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch.optim import Adam, SGD, lr_scheduler
from torchsampler import ImbalancedDatasetSampler

# TorchVision Libraries
import torchvision
from torchvision import models, datasets, transforms
from torchvision.models import resnet50
from PIL import Image

# Metrics
from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score, hamming_loss, matthews_corrcoef, 
    balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
)

# Custom Imports
from elpv_reader import load_dataset
# Environment Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# Transform the label from probability noted in "label.csv" and separate the images in terms of monocrystalline and polycrystalline PV modules. 

# In[5]:


# label transformation - 0: functional; 1: Mildly Defective 2: Moderately Defective 3: Severely Defective
images, proba, types = load_dataset()
def transform_multi(x):
    if x == 1.0:
        return 3
    elif x == 2/3:
        return 2
    elif x == 1/3:
        return 1
    else:
        return 0
label = [transform_multi(i) for i in proba]
mono_index = [i for i, t in enumerate(types) if t == 'mono']
poly_index = [i for i, t in enumerate(types) if t != 'mono']
label_mono = [label[i] for i in mono_index]
label_poly = [label[i] for i in poly_index]
images_mono = [images[i] for i in mono_index]
images_poly = [images[i] for i in poly_index]
mono_data = {
    'mono_index_np': mono_index,
    'y': label_mono}
df = pd.DataFrame(mono_data)


# In[7]:


# Initializes the dataset object with data, targets, and optional transformations
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):       
#         Args:
#             data (numpy.ndarray): The data array, expected to be a numpy array of images.
#             targets (numpy.ndarray or list): The target labels, should be compatible with torch.LongTensor.
#             transform (callable, optional): Optional transform to be applied to each sample.     
        self.data = data
        self.targets =  torch.Tensor(targets)
        self.transform = transform 
    def __getitem__(self, index):
#         Fetches the data and target at a given index and applies transformations if specified.
#         Args:
#             index (int): The index of the data point to be fetched.
#         Returns:
#             tuple: (sample, target) where sample is the transformed data and target is the corresponding label
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        return x, y 
    def __len__(self):
#         Returns the total number of data points in the dataset.
#         Returns:
#             int: The length of the dataset.
        return len(self.data)

def complement(df_full, df_current):
    df_complement = df_full[df_full.mono_index_np.isin(df_current.mono_index_np) == False]
    return df_complement

def train_test_split_straitifed_sampling(random_seed,  images, label):
#     Splits the dataset into train and test sets using stratified sampling based on severity labels.
#     Args:
#         random_seed (int): Random seed for reproducibility.
#         images (list or array-like): List of image data.
#         label (list or array-like): List of labels corresponding to the images.
#     Returns:
#         tuple: Four lists containing the training images, training labels, validation images, and validation labels
    mono_lab = np.array(df['y'])
    mono_severe =df.iloc[np.where(mono_lab == 3)]
    mono_moderate = df.iloc[np.where(mono_lab == 2)]
    mono_mild = df.iloc[np.where(mono_lab == 1)]
    mono_functional = df.iloc[np.where(mono_lab == 0)]
    test_severe = mono_severe.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_severe = complement(mono_severe, test_severe)
    test_moderate = mono_moderate.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_moderate = complement(mono_moderate, test_moderate)
    test_mild =  mono_mild.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_mild =  complement(mono_mild, test_mild)
    test_functional = mono_functional.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_functional = complement(mono_functional, test_functional)
    train_df = pd.concat([train_severe, train_moderate, train_mild, train_functional])
    # shuffle the DataFrame rows
    train_df = train_df.sample(frac = 1)
    test_df = pd.concat([test_severe, test_moderate, test_mild, test_functional])
    test_df = test_df.sample(frac = 1)
    train_indices = train_df['mono_index_np']
    images_train = [images[i] for i in train_indices]
    label_train = [label[i] for i in train_indices]
    test_indices = test_df['mono_index_np']
    images_val = [images[i] for i in test_indices]
    label_val = [label[i] for i in test_indices]
    return images_train, label_train, images_val, label_val


# Created an augmented dataset 
def custom_augmentation(data, label):
    # Define the original transformation: grayscale to RGB images, tensor conversion, and normalization
    transforms_original= torchvision.transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    # Define the horizontal flip transformation with color jitter
    transforms_horizontal_flip = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])  
    ])
    # Define the vertical flip transformation
    transforms_vertical_flip = torchvision.transforms.Compose([
      torchvision.transforms.RandomVerticalFlip(),
     transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    # Define the 90-degree rotation transformation
    transforms_rotate_90 = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    # Define the 180-degree rotation transformation
    transforms_rotate_180 = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180, resample=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    # Define the 270-degree rotation transformation
    transforms_rotate_270 = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(270, resample=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    dat_original = MyDataset(data, label, transform = transforms_original)
    dat_horizontal_flip = MyDataset(data, label, transform = transforms_horizontal_flip)
    dat_vertical_flip = MyDataset(data, label, transform = transforms_vertical_flip)
    dat_rotate_90 = MyDataset(data, label, transform = transforms_rotate_90)
    dat_rotate_180 = MyDataset(data, label, transform = transforms_rotate_180)
    dat_rotate_270 = MyDataset(data, label, transform = transforms_rotate_270)
    # Concatenate all the augmented datasets into one training dataset
    train_data = torch.utils.data.ConcatDataset([dat_original, dat_horizontal_flip, dat_vertical_flip, dat_rotate_90,
                                                dat_rotate_180, dat_rotate_270])
    return train_data



def dataloaders(random_seed, valid_split, dat_size, images, label):
#     Creates data loaders for training and validation datasets with data augmentation.
#     Args:
#         random_seed (int): Random seed for reproducibility.
#         valid_split (float): Fraction of the dataset to use for validation.
#         dat_size (int): Total size of the dataset.
#         images (list or array-like): List of image data.
#         label (list or array-like): List of labels corresponding to the images.

#     Returns:
#         tuple: A dictionary containing the training and validation DataLoader objects,
#                and a dictionary with the sizes of the training and validation datasets 
    split_result = train_test_split(random_seed, images, label)
    images_train = split_result[0]
    label_train = split_result[1]
    images_val = split_result[2]
    label_val = split_result[3]
    train_data = custom_augmentation(images_train, label_train)
    transform_original= torchvision.transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    val_data = MyDataset(images_val, label_val, transform=transform_original)
    train_loader = DataLoader(train_data, batch_size = 64,shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)
    dataloaders= {'train':train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(label_train)*6, 'val': len(label_val)}
    return dataloaders, dataset_sizes 


# In[ ]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the model
def model_vgg19():
#     Creates a modified VGG19 model with a custom classifier for a specific classification task.
#     Returns:
#         nn.Module: A modified VGG19 model
    # Load a pre-trained VGG19 model
    model = torchvision.models.vgg19(pretrained=True)
    # Replace the classifier (fully connected) layer of the VGG19 model
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),  
        nn.ReLU(True),        
        nn.Linear(4096, 2048), 
        nn.ReLU(True),         
        nn.Linear(2048, 1024), 
        nn.ReLU(True),        
        nn.Linear(1024, 4)    
    )
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    ct = 0 
    for child in model.children():
        ct += 1 
        if ct < 3:
            for param in child.parameters():
                param.requires_grad = False  
        elif ct == 3:
            for param in child.parameters():
                param.requires_grad = True  
    return model  
model = model_vgg19()

# train a neural network model while tracking and updating the best-performing model based on validation accuracy 
# return the training and validation loss, accuracy over epochs, and the best model
use_gpu = torch.cuda.is_available()
def train_model(dataloaders, data_sizes, net, criterion, optimizer, scheduler, num_epochs):
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_acc = {}
    y_acc['train'] = []
    y_acc['val'] = []
    x_epoch = []
    since = time.time()
    best_model_wts = net.state_dict()
    best_acc = 0.0
    train_acc_list = []
    train_loss_list = []
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
                outputs = torch.nn.functional.softmax(outputs, dim = 1)
                _, preds = torch.max(outputs.data, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects /data_sizes[phase]
            y_loss[phase].append(epoch_loss)
            y_acc[phase].append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = net.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    net.load_state_dict(best_model_wts)
    return (y_loss['train'], y_loss['val'],  y_acc['train'],  y_acc['val'], net)


def dataloaders(random_seed, valid_split, dat_size, images, label):
    split_result = train_test_split_straitifed_sampling(random_seed,  images, label)
    images_train = split_result[0]
    label_train = split_result[1]
    images_val = split_result[2]
    label_val = split_result[3]
    train_data = custom_augmentation(images_train, label_train)
    transform_original= torchvision.transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    val_data = MyDataset(images_val, label_val, transform = transform_original)
    train_loader = DataLoader(train_data, batch_size = 64,shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)
    dataloaders= {'train':train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(label_train)*6, 'val': len(label_val)}
    return dataloaders, dataset_sizes 

# data_preparation 
mono_dataset = dataloaders(random_seed = 128,  valid_split = 0.2, dat_size = len(mono_index), images = images, label = label)
dataloaders_mono = mono_dataset[0]
dataset_size = mono_dataset[1]
criterions = nn.CrossEntropyLoss()


# Give predictions on a validation dataset using a given model 
# Return the true and predicted class labels

def pred_output(dataloader, model):
    true_list = []
    pred_list = []
    outputs_list = []
    for i, (inputs, classes) in enumerate(dataloader['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        true = classes.tolist()
        true_list.append(true)
        outputs = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim = 1)
        _, preds = torch.max(outputs, 1)
        pred = preds.tolist()
        pred_list.append(pred)
    y_true = list(chain(*true_list))
    y_pred = list(chain(*pred_list))
    result = dict()
    result['pred_list'] = y_pred
    result['true_list'] = y_true
    return result


# train a VGG19 model in two stages
# track the loss and accuracy during training
# return the final model's predictions along with the recorded metrics
def result_output(dataset_loader):
    LOSS = {}
    ACC = {}
    model_arc = model_vgg19()
    model_arc = model_arc.to(device)
    # Observe that all parameters are being optimized
    optimizer_ft1 = optim.SGD(model_arc.parameters(), lr=0.01, momentum = 0.9)
# Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_1 = lr_scheduler.StepLR(optimizer_ft1, step_size = 7, gamma = 0.09)
    train_stage_1 = train_model(dataloaders = dataset_loader, data_sizes = dataset_size, net = model_arc, criterion = criterions, optimizer = optimizer_ft1, scheduler = exp_lr_scheduler_1, num_epochs = 50)
    LOSS['train_stage_1'] = train_stage_1[0]
    LOSS['val_stage_1'] = train_stage_1[1]
    ACC['train_stage_1'] = train_stage_1[2]
    ACC['val_stage_1'] = train_stage_1[3]
    model_ft = train_stage_1[4]
    model_ft = model_ft.to(device)
    optimizer_ft2 = optim.SGD(model_ft.parameters(), lr = 0.005, momentum=0.9)
    exp_lr_scheduler_2 = lr_scheduler.StepLR(optimizer_ft2, step_size = 7, gamma = 0.09)
    ct = 0
    for child in model_ft.children():
        ct = ct + 1
        if ct < 3:
            for param in child.parameters():
                param.require_grad = True 
        elif ct == 3:
            for  param in child.parameters():
                param.require_grad = True
    train_stage_2 = train_model(dataloaders = dataset_loader, data_sizes = dataset_size, net = model_ft, criterion = criterions, optimizer = optimizer_ft2, scheduler = exp_lr_scheduler_2, num_epochs = 50)
    model_ft_2 = train_stage_2[4]
    model_ft_2 = model_ft_2.to(device)
    LOSS['train_stage_2'] = train_stage_2[0]
    LOSS['val_stage_2'] = train_stage_2[1]
    ACC['train_stage_2'] = train_stage_2[2]
    ACC['val_stage_2'] = train_stage_2[3]
    pred_result = pred_output(dataloader = dataset_loader, model = model_ft_2)
    return(LOSS, ACC, pred_result)


# training the model multiple times with different train-test split to evaluate model robustness 

def replicate_pred_output(ran_seed):
    LOSS_REP_TRAIN_S1 = []
    LOSS_REP_TRAIN_S2 = []
    LOSS_REP_VAL_S1 = []
    LOSS_REP_VAL_S2 = []
    ACC_REP_TRAIN_S1 = []
    ACC_REP_TRAIN_S2 = []
    ACC_REP_VAL_S1 = []
    ACC_REP_VAL_S2 = []
    PRED_LIST = []
    for i in ran_seed:
        mono_dataset = dataloaders(random_seed = i,  valid_split = 0.2, dat_size = len(mono_index), images = images, label = label)
        dataloaders_mono = mono_dataset[0]
        dataset_size = mono_dataset[1]
        loss,accuracy, prediction_result = result_output(dataset_loader = dataloaders_mono)
        LOSS_REP_TRAIN_S1.append(loss['train_stage_1'])
        LOSS_REP_TRAIN_S2.append(loss['train_stage_2'])
        LOSS_REP_VAL_S1.append(loss['val_stage_1'])
        LOSS_REP_VAL_S2.append(loss['val_stage_2'])
        ACC_REP_TRAIN_S1.append(accuracy['train_stage_1'])
        ACC_REP_TRAIN_S2.append(accuracy['train_stage_2'])
        ACC_REP_VAL_S1.append(accuracy['val_stage_1'])
        ACC_REP_VAL_S2.append(accuracy['val_stage_2'])
        PRED_LIST.append(prediction_result)
    result = {}
    result['loss_train_s1'] = LOSS_REP_TRAIN_S1
    result['loss_train_s2'] = LOSS_REP_TRAIN_S2
    result['loss_val_s1'] = LOSS_REP_VAL_S1
    result['loss_val_s2'] = LOSS_REP_VAL_S2
    result['acc_train_s1'] = ACC_REP_TRAIN_S1
    result['acc_train_s2'] = ACC_REP_TRAIN_S2
    result['acc_val_s1'] = ACC_REP_VAL_S1
    result['acc_val_s2'] = ACC_REP_VAL_S2
    result['pred_output'] =  PRED_LIST
    return result



random_seed = range(0,50)
result = replicate_pred_output(ran_seed = random_seed)
# save the result
w = csv.writer(open("mono_vgg19_pred.csv", "w"))
for key, val in result.items():
    w.writerow([key, val])

