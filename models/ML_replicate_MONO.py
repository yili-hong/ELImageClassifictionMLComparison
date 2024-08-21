#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Core Libraries
import os
import time
import glob
import csv
import copy
import math
import pathlib
from collections import Counter

# Data Manipulation
import numpy as np
import pandas as pd
from pandas import read_csv

# Image Processing
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from pylab import savefig, rcParams

# Scikit-learn Libraries
from sklearn.model_selection import (
    GridSearchCV, train_test_split, cross_val_score, RepeatedStratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score, hamming_loss,
    matthews_corrcoef, balanced_accuracy_score, precision_score,
    recall_score, f1_score, make_scorer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

# Deep Learning Libraries (Keras)
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
)
from keras.applications import imagenet_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# PyTorch Libraries
import torch
from torch import nn, flatten
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch.optim import Adam, SGD, lr_scheduler
from torchsampler import ImbalancedDatasetSampler
from torchvision import models, datasets, transforms
import torchvision
from torchsummary import summary

# Custom Imports
from elpv_reader import load_dataset
from hypopt import GridSearch

# Set Seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Environment Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def transform_multi(x):
    return {1.0: 3, 2/3: 2, 1/3: 1}.get(x, 0)

# Applying transformation to each probability in the list
labels = [transform_multi(prob) for prob in proba]

# Identifying indices for 'mono' and 'poly' types
mono_indices = [i for i, type_ in enumerate(types) if type_ == 'mono']
poly_indices = [i for i, type_ in enumerate(types) if type_ != 'mono']

# Creating lists for mono and poly labels and images based on the indices
labels_mono = [labels[idx] for idx in mono_indices]
labels_poly = [labels[idx] for idx in poly_indices]
images_mono = [images[idx] for idx in mono_indices]
images_poly = [images[idx] for idx in poly_indices]

# Generating file paths for images stored under "mono_poly/mono" and "mono_poly/poly"
mono_RGB = [os.path.join("mono_poly/mono", img) for img in os.listdir("mono_poly/mono")]
poly_RGB = [os.path.join("mono_poly/poly", img) for img in os.listdir("mono_poly/poly")]

mono_index = [int(img.split("/")[-1].split(".png")[0]) for img in mono_RGB]
poly_index = [int(img.split("/")[-1].split(".png")[0]) for img in poly_RGB]
y_mono = []
for i in mono_index:
    y_mono.append(label_mono[i-1])
# Generate features for the mono dataset using a predefined model
mono_x, mono_features, mono_features_flatten = create_features(mono_RGB, model)

#  initializes the pre-trained VGG16 model for feature extraction 
model = VGG16(weights="imagenet", include_top=False)
def create_features(dataset, pre_model):
    x_scratch = []
    for imagePath in dataset:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        x_scratch.append(image)
    x = np.vstack(x_scratch)
    features = pre_model.predict(x)
    # reshape the extracted features into a flattened vector
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x, features, features_flatten


# SVM model

def mono_replicate_function_svm(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(mono_features_flatten, y_mono_pd, test_size=0.2, random_state = i)
        y_train_mono_series = pd.Series(y_train_mono)
        t = y_train_mono_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
  # Initialize the SVM classifier with custom class weights and other hyperparameters
        model_svm = svm.SVC(gamma='scale', class_weight = weights, C = 0.1, kernel='rbf')
        svm_model = model_svm.fit(X_train_mono, y_train_mono)
        y_pred_svm = svm_model.predict(X_test_mono)
        PRED.append(y_pred_svm)
        TRUE.append(y_test_mono)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE 
    return replicate_result


# In[ ]:

# Logistic Regression
def mono_replicate_function_logistic(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(mono_features_flatten, y_mono_pd, test_size=0.2, random_state = i)
        y_train_mono_series = pd.Series(y_train_mono)
        t = y_train_mono_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
# Initialize the Logistic classifier with custom class weights and other hyperparameters
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight = weights, max_iter=1000).fit(X_train_mono, y_train_mono)
        y_pred_log = mul_lr.predict(X_test_mono)
        PRED.append(y_pred_log)
        TRUE.append(y_test_mono)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE 
    return replicate_result
    


# In[ ]:
# Random Forest

def mono_replicate_function_rf(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(mono_features_flatten, y_mono_pd, test_size=0.2, random_state = i)
        y_train_mono_series = pd.Series(y_train_mono)
        t = y_train_mono_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
# Initialize the  Random Forest with custom class weights and other hyperparameters
        clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1, class_weight=weights)
        rm_fit = clf.fit(X_train_mono, y_train_mono)
        y_pred_rf = rm_fit.predict(X_test_mono)
        PRED.append(y_pred_rf)
        TRUE.append(y_test_mono)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE
    return replicate_result


# In[5]:


random_list = [i for i in range(0, 50)]
p_svm_mono = mono_replicate_function_svm(random_list)
print(p_svm_mono)
w = csv.writer(open("mono_svm.csv", "w"))
# loop over dictionary keys and values
for key, val in p_svm_mono.items():
    w.writerow([key, val])
p_log_mono = mono_replicate_function_logistic(random_list)
print(p_log_mono)
w = csv.writer(open("mono_log.csv", "w"))
# loop over dictionary keys and values
for key, val in p_log_mono.items():
    w.writerow([key, val])
p_rf_mono = mono_replicate_function_logistic(random_list)
print(p_rf_mono)
w = csv.writer(open("mono_rf.csv", "w"))
# loop over dictionary keys and values
for key, val in p_rf_mono.items():
    w.writerow([key, val])

