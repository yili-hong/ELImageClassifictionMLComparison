#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score, hamming_loss,
    matthews_corrcoef, balanced_accuracy_score, precision_score,
    recall_score, f1_score, make_scorer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

# HypOpt Library for Grid Search
from hypopt import GridSearch

# Deep Learning Libraries (Keras and TensorFlow)
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

# Set Seaborn style
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Environment Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[5]:

# Data Loading
from elpv_reader import load_dataset
images, proba, types = load_dataset()
# label transformation
def transform_multi(x):
    if x == 1.0:
        t = 3
    elif x == 2/3:
        t = 2
    elif x == 1/3:
        t = 1
    else:
        t = 0  
    return t
label = []
for i in proba:
    label.append(transform_multi(i))

mono_index = []
poly_index = []
for i in range(len(types)):
    if types[i] == 'mono':
        mono_index.append(i)
    else:
        poly_index.append(i)

label_mono = [label[index] for index in mono_index]
label_poly = [label[index] for index in poly_index]
images_mono = [images[i] for i in mono_index]
images_poly = [images[i] for i in poly_index]
mono_RGB = [os.path.join("mono_poly/mono",img) for img in os.listdir("mono_poly/mono")]
poly_RGB = [os.path.join("mono_poly/poly",img) for img in os.listdir('mono_poly/poly')]


# Read the images 

mono_index = [int(img.split("/")[-1].split(".png")[0]) for img in mono_RGB]
poly_index = [int(img.split("/")[-1].split(".png")[0]) for img in poly_RGB]
y_mono = []
for i in mono_index:
    y_mono.append(label_mono[i-1])
y_poly = []
for j in poly_index:
    y_poly.append(label_poly[j-1])


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
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x, features, features_flatten


# Convert y_poly to pandas Series and count the values for each class
y_poly_pd = pd.Series(y_poly)
class_counts = y_poly_pd.value_counts()
# Generate features for the poly dataset using a predefined model
poly_x, poly_features, poly_features_flatten = create_features(poly_RGB, model)

def complement(df_full, df_current):
    df_complement = df_full[df_full.poly_index_np.isin(df_current.poly_index_np) == False]
    return df_complement

def train_test_split_straitifed_sampling(random_seed):
    poly_lab = np.array(df['y'])
    poly_severe =df.iloc[np.where(poly_lab == 3)]
    poly_moderate = df.iloc[np.where(poly_lab == 2)]
    poly_mild = df.iloc[np.where(poly_lab == 1)]
    poly_functional = df.iloc[np.where(poly_lab == 0)]
    test_severe = poly_severe.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_severe = complement(poly_severe, test_severe)
    test_moderate = poly_moderate.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_moderate = complement(poly_moderate, test_moderate)
    test_mild =  poly_mild.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_mild =  complement(poly_mild, test_mild)
    test_functional = poly_functional.sample(frac = 0.2 , random_state=6).reset_index(drop = True)
    train_functional = complement(poly_functional, test_functional)
    train_df = pd.concat([train_severe, train_moderate, train_mild, train_functional])
    # shuffle the DataFrame rows
    train_df = train_df.sample(frac = 1)
    test_df = pd.concat([test_severe, test_moderate, test_mild, test_functional])
    test_df = test_df.sample(frac = 1)
    train_indices = train_df['poly_index_np']
    test_indices = test_df['poly_index_np']
    return train_df, test_df

# SVM model
def poly_replicate_function_svm(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(poly_features_flatten, y_poly_pd, test_size=0.2, random_state = i)
        y_train_poly_series = pd.Series(y_train_poly)
        t = y_train_poly_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
        model_svm = svm.SVC(gamma='scale', class_weight = weights, C = 0.1, kernel='rbf')
        svm_model = model_svm.fit(X_train_poly, y_train_poly)
        y_pred_svm = svm_model.predict(X_test_poly)
        y_test_poly_list = y_test_poly.tolist()
        PRED.append(y_pred_svm)
        TRUE.append(y_test_poly_list)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE 
    return replicate_result

# Logistic Regression
def poly_replicate_function_logistic(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(poly_features_flatten, y_poly_pd, test_size=0.2, random_state = i)
        y_train_poly_series = pd.Series(y_train_poly)
        t = y_train_poly_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight = weights, max_iter=1000).fit(X_train_poly, y_train_poly)
        y_pred_log = mul_lr.predict(X_test_poly)
        y_test_poly_list = y_test_poly.tolist()
        PRED.append(y_pred_log)
        TRUE.append(y_test_poly_list)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE 
    return replicate_result

# Random Forest
def poly_replicate_function_rf(RANDOM):
    PRED = []
    TRUE = []
    for i in RANDOM: 
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(poly_features_flatten, y_poly_pd, test_size=0.2, random_state = i)
        y_train_poly_series = pd.Series(y_train_poly)
        t = y_train_poly_series.value_counts()
        weights =  {0:t[0], 1:t[1], 2:t[2], 3:t[3]}
        clf = RandomForestClassifier(n_estimators=10000, random_state = 0, n_jobs = -1, class_weight = weights)
        rm_fit = clf.fit(X_train_poly, y_train_poly)
        y_pred_rf = rm_fit.predict(X_test_poly)
        y_test_poly_list = y_test_poly.tolist()
        PRED.append(y_pred_rf)
        TRUE.append(y_test_poly_list)
    replicate_result = dict()
    replicate_result['pred'] = PRED
    replicate_result['true'] = TRUE
    return replicate_result


# In[5]:


random_list = [i for i in range(0, 50)]
p_svm_poly = poly_replicate_function_svm(random_list)
w = csv.writer(open("poly_svm.csv", "w"))
# loop over dictionary keys and values
for key, val in p_svm_poly.items():
    w.writerow([key, val])

# In[ ]:

p_log_poly = poly_replicate_function_logistic(random_list)
w = csv.writer(open("poly_log.csv", "w"))
for key, val in p_log_poly.items():
    w.writerow([key, val])

p_rf_poly = poly_replicate_function_logistic(random_list)
w = csv.writer(open("poly_rf.csv", "w"))
for key, val in p_rf_poly.items():
    w.writerow([key, val])

