#!/usr/bin/env python
# coding: utf-8
# Core libraries
import os
import time
import glob
import copy
import csv
from collections import Counter
from itertools import chain
# Data manipulation
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig, rcParams
# PyTorch libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from torch.optim import Adam, SGD, lr_scheduler
from torchvision import models, datasets, transforms
# Additional PyTorch utilities
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm
# PIL for image handling
from PIL import Image
# Metrics
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, hamming_loss, 
                             matthews_corrcoef, balanced_accuracy_score, 
                             precision_score, recall_score, f1_score, accuracy_score)

# Custom imports
from elpv_reader import load_dataset
# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

# Initializes the dataset object with data, targets, and optional transformations
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
//         """
//         Initializes the dataset object with data, targets, and optional transformations.
//         Args:
//             data (numpy.ndarray): The data array, expected to be a numpy array of images.
//             targets (numpy.ndarray or list): The target labels, should be compatible with torch.LongTensor.
//             transform (callable, optional): Optional transform to be applied to each sample.
//         """
        self.data = data
        self.targets =  torch.Tensor(targets)
        self.transform = transform
    def __getitem__(self, index):
        """
        Fetches the data and target at a given index and applies transformations if specified.
        Args:
            index (int): The index of the data point to be fetched.
        Returns:
            tuple: (sample, target) where sample is the transformed data and target is the corresponding label.
        """
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        return x, y 
    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.data)


# Created an augmented dataset 
def complement(df_full, df_current):
//      """
//     Computes the complement of two DataFrames based on the 'mono_index_np' column.
//     Args:
//         df_full (pd.DataFrame): The complete DataFrame containing all data.
//         df_current (pd.DataFrame): The current DataFrame containing a subset of data.
//     Returns:
//         pd.DataFrame: A DataFrame containing the rows from df_full that are not in df_current 
//                       based on the 'mono_index_np' column.
//     """
    df_complement = df_full[df_full.mono_index_np.isin(df_current.mono_index_np) == False]
    return df_complement

def train_test_split_straitifed_sampling(random_seed, images, label):
//     """
//     Splits the dataset into train and test sets using stratified sampling based on severity labels.
//     Args:
//         random_seed (int): Random seed for reproducibility.
//         images (list or array-like): List of image data.
//         label (list or array-like): List of labels corresponding to the images.
//     Returns:
//         tuple: Four lists containing the training images, training labels, validation images, and validation labels.
//     """
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
    transforms_original= torchvision.transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])

    transforms_horizontal_flip = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])  
    ])

    transforms_vertical_flip = torchvision.transforms.Compose([
      torchvision.transforms.RandomVerticalFlip(),
     transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])

    transforms_rotate_90 = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    
    transforms_rotate_180 = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180, resample=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5968, 0.5968, 0.5968],std=[0.1637, 0.1628, 0.1620])
    ])
    
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
    train_data = torch.utils.data.ConcatDataset([dat_original, dat_horizontal_flip, dat_vertical_flip, dat_rotate_90,
                                                dat_rotate_180, dat_rotate_270])
    return train_data



def dataloaders(random_seed, valid_split, dat_size, images, label):
//     """
//     Creates data loaders for training and validation datasets with data augmentation.

//     Args:
//         random_seed (int): Random seed for reproducibility.
//         valid_split (float): Fraction of the dataset to use for validation.
//         dat_size (int): Total size of the dataset.
//         images (list or array-like): List of image data.
//         label (list or array-like): List of labels corresponding to the images.

//     Returns:
//         tuple: A dictionary containing the training and validation DataLoader objects,
//                and a dictionary with the sizes of the training and validation datasets.
//     """
    split_result = train_test_split_straitifed_sampling(random_seed, images, label)
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
def model_resnet50():
//      """
//     Creates a modified ResNet-50 model with a custom classifier for a specific classification task.
//     Returns:
//         nn.Module: A modified ResNet-50 model.
//     """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)
    model.classifier = nn.Sequential(
    nn.Linear(2048,4096),
    nn.ReLU(True),
    nn.Linear(4096, 2048),
    nn.ReLU(True),
    nn.Linear(2048,1024),
    nn.ReLU(True),
    nn.Linear(1024,4)
    )    
    for name,child in (model.named_children()):
        if name.find('classifier') != -1:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
            
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
    return model 
model = model_resnet50()

# Training function 
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
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0.0
            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
                outputs = torch.nn.functional.softmax(outputs, dim = 1)
                _, preds = torch.max(outputs.data, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
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


# performs a training and evaluation pipeline for a neural network model
def result_output(dataset_loader):
    LOSS = {}
    ACC = {}
    model_arc = model_resnet50()
    model_arc = model_arc.to(device)
    # Observe that all parameters are being optimized
    optimizer_ft1 = optim.SGD(model_arc.parameters(), lr=0.01, momentum = 0.9)
      # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_1 = lr_scheduler.StepLR(optimizer_ft1, step_size = 7, gamma = 0.09)
    train_stage_1 = train_model(dataloaders = dataset_loader, data_sizes = dataset_size, net = model_arc, criterion = criterions, optimizer = optimizer_ft1, scheduler = exp_lr_scheduler_1, num_epochs = 100)
    LOSS['train'] = train_stage_1[0]
    LOSS['val'] = train_stage_1[1]
    ACC['train'] = train_stage_1[2]
    ACC['val'] = train_stage_1[3]
    model_ft = train_stage_1[4]
    pred_result = pred_output(dataloader = dataset_loader, model = model_ft)
    return(LOSS, ACC, pred_result)


# ran_seed: a list of random seeds that will be used for generating different data splits and model initializations
def replicate_pred_output(ran_seed):
    LOSS_REP_TRAIN_S1 = []
    LOSS_REP_VAL_S1 = []
    ACC_REP_TRAIN_S1 = []
    ACC_REP_VAL_S1 = []
    PRED_LIST = []
    for i in ran_seed:
        mono_dataset = dataloaders(random_seed = i,  valid_split = 0.2, dat_size = len(mono_index), images = images, label = label)
        dataloaders_mono = mono_dataset[0]
        dataset_size = mono_dataset[1]
        loss,accuracy, prediction_result = result_output(dataset_loader = dataloaders_mono)
        LOSS_REP_TRAIN_S1.append(loss['train'])
        LOSS_REP_VAL_S1.append(loss['val'])
        ACC_REP_TRAIN_S1.append(accuracy['train'])
        ACC_REP_VAL_S1.append(accuracy['val'])
        PRED_LIST.append(prediction_result)
    result = {}
    result['loss_train'] = LOSS_REP_TRAIN_S1
    result['loss_val'] = LOSS_REP_VAL_S1
    result['acc_train'] = ACC_REP_TRAIN_S1
    result['acc_val'] = ACC_REP_VAL_S1
    result['pred_output'] =  PRED_LIST
    return result


# In[3]:


random_seed = range(0,50)
result = replicate_pred_output(ran_seed = random_seed)
w = csv.writer(open("mono_resnet50_pred.csv", "w"))
# loop over dictionary keys and values
for key, val in result.items():
    w.writerow([key, val])


