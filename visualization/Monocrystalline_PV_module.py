#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard Library Imports
import ast
import itertools
import warnings
warnings.filterwarnings('ignore')
# Third-Party Library Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    hamming_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from matplotlib import rcParams
import os
import csv
# Custom Imports
from elpv_reader import load_dataset


# In[3]:


def string_list(input_string):
    # Convert the input Series into a DataFrame, preserving the original index
    out = pd.DataFrame(input_string.tolist(), index = input_string.index)
    out = out[0].iloc[0]
    output_list = ast.literal_eval(out)
    return output_list

# processing of prediction results stored in a serialized format
def output_pred(input_string):
    PRED = []
    TRUE = []
    out = pd.DataFrame(input_string.tolist(), index = input_string.index)
    out = out[0].iloc[0]
    output_list = ast.literal_eval(out)
    for i in range(len(output_list)):
        PRED.append(output_list[i]['pred_list'])
        TRUE.append(output_list[i]['true_list'])
    pred_result = {}
    pred_result['pred'] = PRED
    pred_result['true'] = TRUE
    return pred_result

# read csv result, extracts predictions and true labels and processes them
def ml_pred_true_split(result_file_name):
    pd_data_frame = pd.read_csv(result_file_name, header = None)
    result_ml = []
    for i in pd_data_frame.iloc:
        result_ml.append(i.loc[1])
    ml_true_result = ast.literal_eval(result_ml[1])
    ml_pred = result_ml[0]
    ml_pred_result = ml_pred_process(ml_pred)
    return (ml_true_result, ml_pred_result)

# Compute performance metrics
def metrics_calculation_overall(y_pred, y_true):
    result = {}
    result['acc'] = accuracy_score(y_true, y_pred)
    result['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    result['hm_loss'] = hamming_loss(y_true, y_pred)
    result['recall'] = recall_score(y_true, y_pred, average = 'macro')
    result['mcc'] = matthews_corrcoef(y_true, y_pred)
    result['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    result['precision'] = precision_score(y_true, y_pred, average = 'macro')
    result['f1_macro'] = f1_score(y_true, y_pred, average ='macro')
    return result

# parse a string that encodes nested array structures (results from serialized machine learning predictions) 
# convert it into a Python list 
def ml_pred_process(input_pred):
    a = input_pred.split('[array(')
    l = a[1].replace("\n       ", "")
    c = l.split(' array(')
    result_pred = []
    for i in c:
        p = i.replace('),', '')
        p = p.replace(')]', '')
        result_pred.append(p)
    PRED = []
    for j in result_pred:
        PRED.append(ast.literal_eval(j))
    return PRED 

# Outcomes of multiple iterations from evaluating the Machine Learning models
def ml_metrics_calculation_overall(pred_reps, true_reps):
    final_result = {}
    ACC = []
    CK = []
    HM_LOSS = []
    RECALL = []
    MCC = []
    BALANCED_ACC = []
    PRECISION = []
    F1_MACRO = []
    for i in range(len(pred_reps)):
        pred_result  = metrics_calculation_overall(pred_reps[i], true_reps[i])
        ACC.append(pred_result['acc'])
        CK.append(pred_result['cohen_kappa'])
        HM_LOSS.append(pred_result['hm_loss'])
        RECALL.append(pred_result['recall'])
        MCC.append(pred_result['mcc'])
        BALANCED_ACC.append(pred_result['balanced_acc'])
        PRECISION.append(pred_result['precision'])
        F1_MACRO.append(pred_result['f1_macro'])                          
    final_result['ACC'] = ACC
    final_result['CK'] = CK
    final_result['HM_LOSS'] = HM_LOSS
    final_result['RECALL'] = RECALL
    final_result['MCC']  = MCC
    final_result['BALANCED_ACC'] = BALANCED_ACC
    final_result['PRECISION'] = PRECISION
    final_result['F1_MACRO'] = F1_MACRO
    return final_result


# Outcomes of multiple iterations from evaluating the Deep Learning Models 
def DL_metrics_calculation(data_csv_file):
    data_df = pd.read_csv(data_csv_file, header = None)
    data_output = data_df.T
    output = data_output.drop(0, axis = 0)
    output = output.set_axis(list(data_df.iloc[:,0]), axis=1)
    pred_output = output.pred_output
    pred_true = output_pred(pred_output)
    pred_list = pred_true['pred']
    true_list = pred_true['true']
    final_result = {}
    ACC = []
    CK = []
    HM_LOSS = []
    RECALL = []
    MCC = []
    BALANCED_ACC = []
    PRECISION = []
    F1_MACRO = []
    for i in range(len(pred_list)):
        result = metrics_calculation_overall(pred_list[i], true_list[i])
        ACC.append(result['acc'])
        CK.append(result['cohen_kappa'])
        HM_LOSS.append(result['hm_loss'])
        RECALL.append(result['recall'])
        MCC.append(result['mcc'])
        BALANCED_ACC.append(result['balanced_acc'])
        PRECISION.append(result['precision'])
        F1_MACRO.append(result['f1_macro'])
    final_result['ACC'] = ACC
    final_result['CK'] = CK
    final_result['HM_LOSS']  = HM_LOSS
    final_result['Recall'] = RECALL
    final_result['MCC'] = MCC
    final_result['BALANCED_ACC'] = BALANCED_ACC
    final_result['PRECISION'] = PRECISION
    final_result['f1_macro'] = F1_MACRO
    return final_result


# Standard deviation calculation of metrics among 50 replicates 
def std_calculation(vgg19_result, resnet50_result, log_result, svm_result, rf_result):
    input_metrics = ['ACC', 'CK', 'Recall', 'MCC', 'BALANCED_ACC', 'PRECISION', 'f1_macro']
    std = {}
    ACC_std = pd.concat([pd.DataFrame(vgg19_result['ACC']).std(), 
                         pd.DataFrame(resnet50_result['ACC']).std(), 
                         pd.DataFrame(log_result['ACC']).std(), 
                         pd.DataFrame(svm_result['ACC']).std(), 
                         pd.DataFrame(rf_result['ACC']).std()])
    CK_std = pd.concat([pd.DataFrame(vgg19_result['CK']).std(), 
                         pd.DataFrame(resnet50_result['CK']).std(), 
                         pd.DataFrame(log_result['CK']).std(), 
                         pd.DataFrame(svm_result['CK']).std(), 
                         pd.DataFrame(rf_result['CK']).std()])
    recall_std = pd.concat([pd.DataFrame(vgg19_result['RECALL']).std(), 
                         pd.DataFrame(resnet50_result['RECALL']).std(), 
                         pd.DataFrame(log_result['RECALL']).std(), 
                         pd.DataFrame(svm_result['RECALL']).std(), 
                         pd.DataFrame(rf_result['RECALL']).std()])
    MCC_std = pd.concat([pd.DataFrame(vgg19_result['MCC']).std(), 
                         pd.DataFrame(resnet50_result['MCC']).std(), 
                         pd.DataFrame(log_result['MCC']).std(), 
                         pd.DataFrame(svm_result['MCC']).std(), 
                         pd.DataFrame(rf_result['MCC']).std()])
    bal_acc_std = pd.concat([pd.DataFrame(vgg19_result['BALANCED_ACC']).std(), 
                         pd.DataFrame(resnet50_result['BALANCED_ACC']).std(), 
                         pd.DataFrame(log_result['BALANCED_ACC']).std(), 
                         pd.DataFrame(svm_result['BALANCED_ACC']).std(), 
                         pd.DataFrame(rf_result['BALANCED_ACC']).std()])
    precision_std = pd.concat([pd.DataFrame(vgg19_result['PRECISION']).std(), 
                         pd.DataFrame(resnet50_result['PRECISION']).std(), 
                         pd.DataFrame(log_result['PRECISION']).std(), 
                         pd.DataFrame(svm_result['PRECISION']).std(), 
                         pd.DataFrame(rf_result['PRECISION']).std()])
    f1_macro_std = pd.concat([pd.DataFrame(vgg19_result['F1_MACRO']).std(), 
                         pd.DataFrame(resnet50_result['F1_MACRO']).std(), 
                         pd.DataFrame(log_result['F1_MACRO']).std(), 
                         pd.DataFrame(svm_result['F1_MACRO']).std(), 
                         pd.DataFrame(rf_result['F1_MACRO']).std()])
    std['ACC'] = ACC_std
    std['CK'] = CK_std
    std['recall'] = recall_std
    std['MCC'] = MCC_std
    std['bal_acc_std'] = bal_acc_std
    std['precision'] = precision_std 
    std['f1_macro_std'] = f1_macro_std
    return std

# Median calculation of metrics among 50 replicates 
def median_calculation(vgg19_result, resnet50_result, log_result, svm_result, rf_result):
    input_metrics = ['ACC', 'CK', 'Recall', 'MCC', 'BALANCED_ACC', 'PRECISION', 'f1_macro']
    median = {}
    ACC_median = pd.concat([pd.DataFrame(vgg19_result['ACC']).median(), 
                         pd.DataFrame(resnet50_result['ACC']).median(), 
                         pd.DataFrame(log_result['ACC']).median(), 
                         pd.DataFrame(svm_result['ACC']).median(), 
                         pd.DataFrame(rf_result['ACC']).median()])
    CK_median = pd.concat([pd.DataFrame(vgg19_result['CK']).median(), 
                         pd.DataFrame(resnet50_result['CK']).median(), 
                         pd.DataFrame(log_result['CK']).median(), 
                         pd.DataFrame(svm_result['CK']).median(), 
                         pd.DataFrame(rf_result['CK']).median()])
    recall_median = pd.concat([pd.DataFrame(vgg19_result['RECALL']).median(), 
                         pd.DataFrame(resnet50_result['RECALL']).median(), 
                         pd.DataFrame(log_result['RECALL']).median(), 
                         pd.DataFrame(svm_result['RECALL']).median(), 
                         pd.DataFrame(rf_result['RECALL']).median()])
    MCC_median = pd.concat([pd.DataFrame(vgg19_result['MCC']).median(), 
                         pd.DataFrame(resnet50_result['MCC']).median(), 
                         pd.DataFrame(log_result['MCC']).median(), 
                         pd.DataFrame(svm_result['MCC']).median(), 
                         pd.DataFrame(rf_result['MCC']).median()])
    bal_acc_median = pd.concat([pd.DataFrame(vgg19_result['BALANCED_ACC']).median(), 
                         pd.DataFrame(resnet50_result['BALANCED_ACC']).median(), 
                         pd.DataFrame(log_result['BALANCED_ACC']).median(), 
                         pd.DataFrame(svm_result['BALANCED_ACC']).median(), 
                         pd.DataFrame(rf_result['BALANCED_ACC']).median()])
    precision_median = pd.concat([pd.DataFrame(vgg19_result['PRECISION']).median(), 
                         pd.DataFrame(resnet50_result['PRECISION']).median(), 
                         pd.DataFrame(log_result['PRECISION']).median(), 
                         pd.DataFrame(svm_result['PRECISION']).median(), 
                         pd.DataFrame(rf_result['PRECISION']).median()])
    f1_macro_median = pd.concat([pd.DataFrame(vgg19_result['F1_MACRO']).median(), 
                         pd.DataFrame(resnet50_result['F1_MACRO']).median(), 
                         pd.DataFrame(log_result['F1_MACRO']).median(), 
                         pd.DataFrame(svm_result['F1_MACRO']).median(), 
                         pd.DataFrame(rf_result['F1_MACRO']).median()])
    median['ACC'] = ACC_median
    median['CK'] = CK_median
    median['recall'] = recall_median
    median['MCC'] = MCC_median
    median['bal_acc_std'] = bal_acc_median
    median['precision'] = precision_median
    median['f1_macro_std'] = f1_macro_median
    return median

# Reorgnize the data to prepare the plot
def median_reorganization(median_crystalline_list):
    median_result = []
    for i in range(5):
        model_median_result = list((median_crystalline_list['MCC'].iloc[i],
                                 median_crystalline_list['ACC'].iloc[i], 
                                median_crystalline_list['recall'].iloc[i],
                                median_crystalline_list['precision'].iloc[i],
                                median_crystalline_list['bal_acc_std'].iloc[i],
                                median_crystalline_list['f1_macro_std'].iloc[i]))
        median_result.append(model_median_result)
    median_model = {}
    median_model['VGG19'] = median_result[0]
    median_model['ResNet50'] = median_result[1]
    median_model['Log'] = median_result[2]
    median_model['SVM'] = median_result[3]
    median_model['RF'] = median_result[4]
    return median_model

# Reorgnize the data to prepare the plot
def std_reorganization(std_crystalline_list):
    std_result = []
    for i in range(5):
        model_std_result = list((std_crystalline_list['MCC'].iloc[i],
                                 std_crystalline_list['ACC'].iloc[i], 
                                #std_crystalline_list['recall'].iloc[i],
                                std_crystalline_list['precision'].iloc[i],
                                std_crystalline_list['bal_acc_std'].iloc[i],
                                std_crystalline_list['f1_macro_std'].iloc[i]))
        std_result.append(model_std_result)
    std_model = {}
    std_model['VGG19'] = std_result[0]
    std_model['ResNet50'] = std_result[1]
    std_model['Log'] = std_result[2]
    std_model['SVM'] = std_result[3]
    std_model['RF'] = std_result[4]
    return std_model

# Plot of Loss versus epochs 
def loss_plot(train_loss_list, val_loss_list, epoch_list, plot_title):
    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(7)
    ax.plot(epoch_list, train_loss_list, label='Training loss', color = 'blue')
    ax.plot(epoch_list, val_loss_list, 'k--', label='Validation loss', color = 'black')
    legend = ax.legend(loc='right', shadow=True, fontsize= 12)
    #plt.title(plot_title, fontsize = 15)
    plt.savefig(plot_title)

# Process the CSV file with machine learning evaluation data to extract metrics for one-stage or two-stage training models
def loss_extract(data_csv, which_run, one_stage):
    final_loss = {}
    data_read = pd.read_csv(data_csv, header = None)
    data_transform = data_read.T
    data_trans = data_transform.drop(0, axis = 0)
    data_trans = data_trans.set_axis(list(data_read.iloc[:,0]), axis = 1)
    if one_stage == True:
        loss_train = data_trans.loss_train
        loss_val = data_trans.loss_val
        acc_train = data_trans.acc_train
        acc_val = data_trans.acc_val
        pred_output = data_trans.pred_output
        loss_train_data = string_list(loss_train)
        loss_val_data = string_list(loss_val)
        final_train_loss = loss_train_data[which_run]
        final_val_loss = loss_val_data[which_run]
        final_loss['train_loss'] = final_train_loss
        final_loss['val_loss'] = final_val_loss
    else: 
        loss_train_s1 = string_list(data_trans.loss_train_s1)
        loss_train_s2 = string_list(data_trans.loss_train_s2)
        loss_val_s1 = string_list(data_trans.loss_val_s1)
        loss_val_s2 = string_list(data_trans.loss_val_s2)
        final_train_loss_s1 = loss_train_s1[which_run]
        final_train_loss_s2 = loss_train_s2[which_run]
        final_val_loss_s1 = loss_val_s1[which_run]
        final_val_loss_s2 = loss_val_s2[which_run]
        final_loss['train_loss_s1'] = final_train_loss_s1
        final_loss['train_loss_s2'] = final_train_loss_s2
        final_loss['val_loss_s1'] = final_val_loss_s1
        final_loss['val_loss_s2'] = final_val_loss_s2
    return final_loss

# Density plot
def multiple_metrics_density_plot(metric_list, model, metrics, plot_title):
    metric_list = np.array(metric_list)
    # consider two model: VGG19 and ResNet50 
    model_label = np.repeat(model, [50, 50, 50, 50, 50], axis = 0)
    df = pd.DataFrame({'Model':model_label, metrics: metric_list})
    # convert to wider data frame 
    data_wide = df.pivot(columns = 'Model', values = metrics)
    data_wide.plot.kde(figsize = (7, 7), linewidth = 3)
    plt.xlabel(metrics, size = 15)
    plt.ylabel('density', fontsize = 15)
    plt.legend(fontsize = 15)
    return df 

# Considering multiple jobs submitted and multiple csv files output, combine the results together
def metric_combine(file_name_1, file_name_2, file_name_3, file_name_4, file_name_5):
    final_result = {}
    final_result_1 = DL_metrics_calculation(file_name_1)
    final_result_2 = DL_metrics_calculation(file_name_2)
    final_result_3 = DL_metrics_calculation(file_name_3)
    final_result_4 = DL_metrics_calculation(file_name_4)
    final_result_5 = DL_metrics_calculation(file_name_5)
    accuracy =  list(itertools.chain(final_result_1['ACC'], final_result_2['ACC'], final_result_3['ACC'], final_result_4['ACC'], final_result_5['ACC']))
    CK = list(itertools.chain(final_result_1['CK'], final_result_2['CK'], final_result_3['CK'], final_result_4['CK'], final_result_5['CK']))
    HM_LOSS = list(itertools.chain(final_result_1['HM_LOSS'], final_result_2['HM_LOSS'], final_result_3['HM_LOSS'], final_result_4['HM_LOSS'], final_result_5['HM_LOSS']))
    Recall = list(itertools.chain(final_result_1['Recall'], final_result_2['Recall'], final_result_3['Recall'], final_result_4['Recall'],final_result_5['Recall']))
    MCC = list(itertools.chain(final_result_1['MCC'], final_result_2['MCC'],final_result_3['MCC'], final_result_4['MCC'], final_result_5['MCC']))
    BALANCED_ACC = list(itertools.chain(final_result_1['BALANCED_ACC'], final_result_2['BALANCED_ACC'], final_result_3['BALANCED_ACC'], final_result_4['BALANCED_ACC'], final_result_5['BALANCED_ACC']))
    PRECISION = list(itertools.chain(final_result_1['PRECISION'], final_result_2['PRECISION'], final_result_3['PRECISION'], final_result_4['PRECISION'], final_result_5['PRECISION']))
    F1_MACRO = list(itertools.chain(final_result_1['f1_macro'], final_result_2['f1_macro'], final_result_3['f1_macro'], final_result_4['f1_macro'], final_result_5['f1_macro']))
    final_result['ACC'] = accuracy
    final_result['CK'] = CK
    final_result['HM_LOSS']  = HM_LOSS
    final_result['RECALL'] = Recall
    final_result['MCC'] = MCC
    final_result['BALANCED_ACC'] = BALANCED_ACC
    final_result['PRECISION'] = PRECISION
    final_result['F1_MACRO'] = F1_MACRO
    return final_result
    
def confusion_matrix_plot(true_list, pred_list, cm_name):
    cm_df = confusion_matrix(true_list, pred_list)
    cm_df = pd.DataFrame(cm_df,
                     index = ['Functional','Mild D','Moderate D', 'Severe D'], 
                     columns = ['Functional','Mild D','Moderate D', 'Severe D'])
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, annot = True, fmt='g', cmap = 'Blues' )
    sns.set(font_scale=1.4)
   # plt.title(cm_name)
    plt.xticks(rotation = 15)
    plt.xlabel('xlabel', fontsize = 15)
    plt.ylabel('ylabel', fontsize = 15)
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(cm_name)
    plt.show()   

# One model runs 51 replicates, one more than expected and remove extra one (selected randomly) for fairness of std comparison
def delete_one_obs(dict_list, delete_index):
    new_dict = {}
    acc = dict_list['ACC']
    del acc[delete_index]
    CK = dict_list['CK']
    del CK[delete_index]
    hm_loss = dict_list['HM_LOSS']
    del hm_loss[delete_index]
    Recall = dict_list['RECALL']
    del Recall[delete_index]
    mcc = dict_list['MCC']
    del mcc[delete_index]
    BALANCED_ACC = dict_list['BALANCED_ACC']
    del BALANCED_ACC[delete_index]
    precision = dict_list['PRECISION']
    del precision[delete_index]
    f1_macro = dict_list['F1_MACRO']
    del f1_macro[delete_index]
    new_dict['ACC'] = acc
    new_dict['CK'] = CK
    new_dict['HM_LOSS']  = hm_loss
    new_dict['RECALL'] = Recall
    new_dict['MCC'] = mcc
    new_dict['BALANCED_ACC'] = BALANCED_ACC
    new_dict['PRECISION'] = precision
    new_dict['F1_MACRO'] = f1_macro
    return new_dict

# restructures evaluation metrics for multiple machine learning models from a single dictionary into an organized format
def model_dividence_bst_model(input_bst_dict):
   # model_sequence = ['VGG19', 'ResNet50', 'Log', 'SVM', 'Random Forest']
    output_bst_model_result = {}
    VGG19 = (input_bst_dict['mcc'][0], input_bst_dict['balanced_acc'][0], #input_bst_dict['recall'][0],
                        input_bst_dict['precision'][0], input_bst_dict['accuracy'][0], input_bst_dict['f1_macro'][0])
    
    ResNet50 = (input_bst_dict['mcc'][1], input_bst_dict['balanced_acc'][1], #input_bst_dict['recall'][1],
                        input_bst_dict['precision'][1], input_bst_dict['accuracy'][1], input_bst_dict['f1_macro'][1])
    
    Log = (input_bst_dict['mcc'][2], input_bst_dict['balanced_acc'][2], #input_bst_dict['recall'][2],
                        input_bst_dict['precision'][2], input_bst_dict['accuracy'][2], input_bst_dict['f1_macro'][2])
    
    SVM = (input_bst_dict['mcc'][3], input_bst_dict['balanced_acc'][3], #input_bst_dict['recall'][3],
                        input_bst_dict['precision'][3], input_bst_dict['accuracy'][3], input_bst_dict['f1_macro'][3])
    
    RF = (input_bst_dict['mcc'][4], input_bst_dict['balanced_acc'][4], #input_bst_dict['recall'][4],
                        input_bst_dict['precision'][4], input_bst_dict['accuracy'][4], input_bst_dict['f1_macro'][4])
    
    output_bst_model_result['VGG19'] = list(VGG19)
    output_bst_model_result['ResNet50'] = list(ResNet50)
    output_bst_model_result['Log'] = list(Log)
    output_bst_model_result['SVM'] = list(SVM)
    output_bst_model_result['RF'] = list(RF)
    return output_bst_model_result


# In[4]:
# Reorg the dataset 
def median_format_reorganize_data(input_median_metric):
    category = ['mcc', 'accuracy', 'precision', 'balanced_acc', 'f1_macro']
    VGG_19 = (input_median_metric[category[0]][0], input_median_metric[category[1]][0],
                 input_median_metric[category[2]][0], input_median_metric[category[3]][0],
                 input_median_metric[category[4]][0]) 
    ResNet50 = (input_median_metric[category[0]][1], input_median_metric[category[1]][1],
                 input_median_metric[category[2]][1], input_median_metric[category[3]][1],
                 input_median_metric[category[4]][1]) 
    Log = (input_median_metric[category[0]][2], input_median_metric[category[1]][2],
                 input_median_metric[category[2]][2], input_median_metric[category[3]][2],
                 input_median_metric[category[4]][2]) 
    SVM = (input_median_metric[category[0]][3], input_median_metric[category[1]][3],
                 input_median_metric[category[2]][3], input_median_metric[category[3]][3],
                 input_median_metric[category[4]][3]) 
    RF = (input_median_metric[category[0]][4], input_median_metric[category[1]][4],
                 input_median_metric[category[2]][4], input_median_metric[category[3]][4],
                 input_median_metric[category[4]][4]) 
    result = {}
    result['VGG19'] = list(VGG_19)
    result['ResNet50'] = list(ResNet50)
    result['Log'] = list(Log)
    result['SVM'] = list(SVM)
    result['RF'] = list(RF)
    return result
# Reorg the data format
def median_metrics(vgg19_bst, resnet50_bst, log_bst, svm_bst, rf_bst):
    bst_result = {}
    acc_bst = list((vgg19_bst[0], resnet50_bst[0], log_bst[0], svm_bst[0], rf_bst[0]))
    ck_bst =  list((vgg19_bst[1], resnet50_bst[1], log_bst[1], svm_bst[1], rf_bst[1]))
    hm_loss_bst = list((vgg19_bst[2], resnet50_bst[2], log_bst[2], svm_bst[2], rf_bst[2]))
    recall_bst = list((vgg19_bst[3], resnet50_bst[3], log_bst[3], svm_bst[3], rf_bst[3]))
    mcc_bst = list((vgg19_bst[4], resnet50_bst[4], log_bst[4], svm_bst[4], rf_bst[4]))
    balanced_acc_bst = list((vgg19_bst[5], resnet50_bst[5], log_bst[5], svm_bst[5], rf_bst[5]))
    precision_bst = list((vgg19_bst[6], resnet50_bst[6], log_bst[6], svm_bst[6], rf_bst[6]))
    f1_macro_bst = list((vgg19_bst[7], resnet50_bst[7], log_bst[7], svm_bst[7], rf_bst[7]))
    bst_result['accuracy'] = acc_bst
    bst_result['ck'] = ck_bst 
    bst_result['hm_loss'] = hm_loss_bst
    bst_result['recall'] = recall_bst
    bst_result['mcc'] = mcc_bst
    bst_result['balanced_acc'] = balanced_acc_bst
    bst_result['precision'] = precision_bst
    bst_result['f1_macro'] = f1_macro_bst
    return bst_result


# In[22]:

# plot the histogram of severity of defectiveness EL images based on monocrystalline PV modules 
def returnSum(myDict):
    list = []
    for i in myDict:
        list.append(myDict[i])
    final = sum(list)
    return final
def plot_bar_with_percentage(data, xlabel_list,  total, figure_size, bar_color, file_name):
    total = returnSum(data)
    percentages = [value / total * 100 for value in data.values()]
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots(figsize = figure_size)
    ax.bar(labels, values, color = bar_color)
    xlabel = xlabel_list
    for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
        ax.text(i, value, f"{percentages[i]:.2f}%", ha='center', va='bottom', fontsize =14, fontname='Times New Roman')
    ax.set_xlabel('Severity of Defectiveness', fontsize = 15,fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontsize = 15, fontname='Times New Roman')
    plt.xticks(y_pos, bars, fontsize = 13, fontname ='Times New Roman')
    plt.yticks(fontsize = 13, fontname = 'Times New Roman')
    plt.savefig(file_name)
    plt.show()
bars = ['Functional', 'Mildly D', 'Moderately D', 'Severely D']
y_pos = np.arange(len(bars))
xlabel = ['Functional', 'Mildly D', 'Moderately D', 'Severely D']
mono_data = {'Functional': 588, 'Mildly D': 117, 'Moderately D':56, 'Severely D': 313}
plot_bar_with_percentage(data = mono_data, xlabel_list = xlabel, total = returnSum(mono_data), figure_size = (7, 7), 
                         bar_color = 'purple', file_name = 'mono_description.pdf')


# In[ ]:

# Read the csv result file of both ML and DL models 
read_path = 'csv_result/' 
os.chdir(read_path)
mono_resnet50_metrics = metric_combine(file_name_1 = 'mono_resnet50_pred010.csv', file_name_2 = 'mono_resnet50_pred1020.csv', 
                                   file_name_3 = 'mono_resnet50_pred2030.csv', file_name_4 = 'mono_resnet50_pred3040.csv',
                                   file_name_5 = 'mono_resnet50_pred4050.csv') 
mono_vgg19_metrics = metric_combine(file_name_1 = 'mono_vgg19_pred010.csv', file_name_2 = 'mono_vgg19_pred1020.csv', 
                                   file_name_3 = 'mono_vgg19_pred2030.csv', file_name_4 = 'mono_vgg19_pred3040.csv',
                                   file_name_5 = 'mono_vgg19_pred4050.csv')
mono_rf = ml_pred_true_split('mono_rf.csv')
mono_log = ml_pred_true_split('mono_log.csv')
mono_svm = ml_pred_true_split('mono_svm.csv')
mono_log_result = ml_metrics_calculation_overall(mono_log[1], mono_log[0])
mono_svm_result = ml_metrics_calculation_overall(mono_svm[1], mono_svm[0])
mono_rf_result = ml_metrics_calculation_overall(mono_rf[1], mono_rf[0])


# In[5]:

# Plot loss of ResNet50 model, 2nd replicates as an example 
resnet50_bst_loss = loss_extract(data_csv = "mono_resnet50_pred1020.csv", which_run = 1, one_stage = True)
loss_plot(resnet50_bst_loss['train_loss'], resnet50_bst_loss['val_loss'], epoch_list = [i+1 for i in range(0, 100)],
         plot_title = 'Loss_Mono_BST_ResNet50.pdf')


# ### Standard Deviation

# In[7]:


std_mono = std_calculation(mono_vgg19_metrics, mono_resnet50_metrics, mono_log_result, mono_svm_result, mono_rf_result)
mono_std = std_reorganization(std_mono)


# In[16]:
# Radar plot of standard deviation of metrics among 50 replicates
categories = ['MCC', 'Accuracy', 'Precision', 'Balanced Accuracy', 'F1-Macro']
categories = [*categories, categories[0]]  # Close the loop
VGG19 = mono_std['VGG19']
ResNet50 = mono_std['ResNet50']
Log = mono_std['Log']
SVM = mono_std['SVM']
RF = mono_std['RF']
VGG19 = [*VGG19, VGG19[0]]
ResNet50 = [*ResNet50, ResNet50[0]]
Log = [*Log, Log[0]]
SVM = [*SVM, SVM[0]]
RF = [*RF, RF[0]]
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(label_loc, VGG19, label='VGG19', linewidth=3, marker='o')
ax.plot(label_loc, ResNet50, label='ResNet50', linewidth=3, marker='p')
ax.plot(label_loc, Log, label='Logit', linewidth=3, marker='*')
ax.plot(label_loc, SVM, label='SVM', linewidth=3, marker='H')
ax.plot(label_loc, RF, label='RF', linewidth=3, marker='D')
ax.set_xticks(label_loc[:-1])  # Exclude the last tick because it's the same as the first
ax.set_xticklabels(categories[:-1], fontsize=15)  # Exclude the repeated category
angles = np.linspace(0, 2 * np.pi, len(categories))
plt.legend(bbox_to_anchor=(1.25, 1), fontsize=14)
plt.savefig("Mono_std.pdf", bbox_inches='tight')
plt.show()


# ### Median Radar plot

# In[17]:


# identify the replications whose accuracy is median value
# median: log - random seed = 49
mono_log_acc = pd.Series(mono_log_result['ACC'])
# median:svm - random seed = 44
mono_svm_acc = pd.Series(mono_svm_result['ACC'])
# median: rf - random seed = 23
mono_rf_acc = pd.Series(mono_rf_result['ACC'])
# median: VGG19 - random seed = 40
pd.Series(mono_vgg19_metrics['ACC'])
# median: resnet50 - random seed = 49
pd.Series(mono_vgg19_metrics['ACC'])
mono_vgg19_median = []
for key in mono_vgg19_metrics.keys():
    mono_vgg19_median.append(mono_vgg19_metrics[key][40])

mono_resnet50_median = []
for key in mono_resnet50_metrics.keys():
    mono_resnet50_median.append(mono_resnet50_metrics[key][49])

mono_log_median = []
for key in mono_log_result.keys():
    mono_log_median.append(mono_log_result[key][49])
    
mono_svm_median = []
for key in mono_svm_result.keys():
    mono_svm_median.append(mono_svm_result[key][44])

mono_rf_median = []
for key in mono_rf_result.keys():
    mono_rf_median.append(mono_rf_result[key][23])


# In[20]:


median_mono = median_metrics(mono_vgg19_median, mono_resnet50_median, mono_log_median, mono_svm_median, mono_rf_median)
median_mono_model = model_dividence_bst_model(median_mono)
mono_med = median_format_reorganize_data(median_mono)

# Radar plot of Median of metrics among 50 replicates

categories =  ['MCC', 'Accuracy', 'Precision', 'Balanced Accuracy', 'F1-Macro']
categories = [*categories, categories[0]]
VGG19  = mono_med['VGG19']
ResNet50 = mono_med['ResNet50']
Log =  mono_med['Log']
SVM =  mono_med['SVM']
RF  = mono_med['RF']
VGG19 = [*VGG19, VGG19[0]]
ResNet50 = [*ResNet50, ResNet50[0]]
Log = [*Log, Log[0]]
SVM = [*SVM, SVM[0]]
RF = [*RF, RF[0]]
label_loc= np.linspace(start = 0, stop = 2 * np.pi, num = len(VGG19))
# label_loc
plt.figure(figsize=(8, 8))
ax = plt.subplot(polar = True)
plt.plot(label_loc, VGG19, label = 'VGG19', linewidth = 3, marker='o')
plt.plot(label_loc, ResNet50, label = 'ResNet50', linewidth = 3, marker = 'p')
plt.plot(label_loc, Log, label = 'Logit', linewidth =3, marker = '*')
plt.plot(label_loc, SVM, label = 'SVM', linewidth =3,  marker = 'H')
plt.plot(label_loc, Log, label = 'RF', linewidth = 3, marker = 'D')
lines,labels = plt.thetagrids(np.degrees(label_loc), labels = categories)
plt.legend(bbox_to_anchor=(1.25,1), fontsize = 14)
i = 0
angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
angles = np.rad2deg(angles)
for label, angle in zip(ax.get_xticklabels(), angles):
    if i <= 5:
        if i == 2 or i == 1:
            i = i + 1
            x,y = label.get_position()
            lab = ax.text(x,y-.08, label.get_text(), transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va(),fontsize = 15)
            labels.append(lab)
        elif i == 3:
            i = i + 1
            x,y = label.get_position()
            lab = ax.text(x,y-.08, label.get_text(), transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va(),fontsize = 15)
            lab.set_rotation(angle + 0)
            labels.append(lab)    
        else:
            x,y = label.get_position()
            lab = ax.text(x,y-.08, label.get_text(), transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va(),fontsize = 15)
            #lab.set_rotation(angle)
            labels.append(lab)
            i = i + 1
ax.set_xticklabels([])
plt.savefig("Mono_med.pdf", bbox_inches = 'tight')
# plt.show()

