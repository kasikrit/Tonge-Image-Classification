#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:29:07 2023

@author: kasikritdamkliang
"""
import tensorflow as tf
from tensorflow import keras
import os, platform
import time
import cv2
import datetime as dt
import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, optimizers, Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
    array_to_img, img_to_array, load_img)
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

from livelossplot import PlotLossesKeras

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imutils import paths

# from keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, GlobalAveragePooling2D,
    BatchNormalization, Dropout, concatenate)
# from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# from keras_applications.resnet import ResNet50

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2, preprocess_input)
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3, preprocess_input)
from tensorflow.keras.applications.densenet import (
    DenseNet121, preprocess_input)
from tensorflow.keras.applications.resnet import (
    ResNet50, ResNet101, ResNet152, preprocess_input)
# from tensorflow.keras.applications.resnet_v2 import (
#     ResNet101V2, ResNet50V2, preprocess_input)

from tensorflow.keras.applications.mobilenet import (
    MobileNet, preprocess_input)

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input)

from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input)

from tensorflow.keras.optimizers import Adadelta, Nadam, Adam

from tqdm import tqdm
from vit_keras import vit, utils, visualize

import random
import json
from yacs.config import CfgNode as CN
import configsEva
config = configsEva.get_config()

# from tensorflow.keras.applications.efficientnet import (
#     EfficientNetB7, preprocess_input)
from tensorflow.keras.applications import (vgg16, vgg19,
    EfficientNetB7, xception, efficientnet, inception_resnet_v2,
    inception_v3, densenet, resnet, resnet_v2, 
    mobilenet, mobilenet_v2)

def cfg_to_dict(cfg_node):
    """
    Recursively convert a YACS CfgNode to a nested dictionary.
    """
    if isinstance(cfg_node, CN):
        return {k: v for k, v in cfg_node.items()}
    return cfg_node

def filter_hidden_files(file_paths):
    """
    Filter out paths that start with a '.' indicating they are hidden.
    This only considers the filename, not part of the path, as hidden.
    """
    return [path for path in file_paths if not os.path.basename(path).startswith('.')]

# def seed_everything(seed = seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

# seed_everything()

print(tf.__version__) #2.11.0, python = 3.9
#2.3.0, python = 3.6
print(tf.test.gpu_device_name())

print(keras.__version__) #2.11.0, python = 3.9
#2.4.0, python = 3.6

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
def make_pair(imgs,labels):
    pairs = []
    for img, mask in zip(imgs, labels):
        pairs.append( (img, mask) )
    
    return pairs

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
   
#%%
if config['DATA']['SET'] == 'C':
    print(config['DATA']['IMAGE'])
    print(config['DATA']['MASK'])
    imagePaths = list(paths.list_images(config['DATA']['IMAGE']))   
    imagePaths = filter_hidden_files(imagePaths)
    print(len(imagePaths))
    
    maskPaths = list(paths.list_images(config['DATA']['MASK']))   
    maskPaths = filter_hidden_files(maskPaths)
    print(len(maskPaths))
    
    imagePaths.sort()
    maskPaths.sort()
    
    maskPaths_filtered = list()
    for path in imagePaths:
        case_id = os.path.basename(path).split('-')[0]
        # print(case_id)
        mask_file = os.path.join(config['DATA']['MASK'],
                    case_id + '-crop_mask.png')
        print(mask_file)
        if mask_file in maskPaths:
            maskPaths_filtered.append(mask_file)
    print(len(maskPaths_filtered))
    
    trainPairs = make_pair(imagePaths, maskPaths_filtered)
    
else:
    print(config['DATA']['TRAIN_PATH'])
    
    trainPaths = list(paths.list_images(config['DATA']['TRAIN_PATH']))   
    trainPaths = filter_hidden_files(trainPaths)
    totalTrain = len(trainPaths)
    print(totalTrain)
    
    trainPaths.sort()
    
    class_dict = dict(config.TRAIN.CLASSDICT)
    print(class_dict)
    
#%%
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
X = list()
X_ID = list()
cnt = 0
for file in tqdm(trainPaths):
    img = load_img(file, target_size=(config['DATA']['W'], config['DATA']['H']))
    X.append(np.array(img))
    X_ID.append(os.path.basename(file).split('_')[0])
    cnt+=1
    # if cnt%20==0: 
    #     print(cnt, X_ID)

X = np.array(X)
#X_train=X_train.reshape(len(X), W, H, C)
#X = X/255.0
print(X.shape)

#%%
# read train 
features = pd.read_csv(config['DATA']['FEATURE_PATH'])
# print(features.shape)
# print(features.head())
if config['DATA']['DEVICE'] == 'DSLR':
    features_filtered = features.loc[features['Camera'] == 0]
else:
    features_filtered = features.loc[features['Camera'] == 1]

features_filtered = features_filtered.dropna(subset=['Class'])

print(features_filtered.shape)
print(features_filtered.head())
        
#%%
class_list = []
for path in trainPaths:
    filename = os.path.basename(path)
    case_id = int(filename.split('_')[0])
       
    # Replace 'your_filename1_value' with the specific Filename1 value you're looking for
    specific_class = features_filtered.loc[features_filtered['ID'] == case_id,
                                          'Class'].iloc[0]
                             
    # print(case_id, specific_class)
    class_list.append(int(specific_class))

print('len(class_list): ', len(class_list))

# import math

# # Assuming 'class_list' is your list that might contain NaN values
# class_list_cleaned = [x for x in class_list if not (isinstance(x, float) 
#                         and math.isnan(x))]

# print('len: ', len(class_list_cleaned))

#%% 
# y = features['Class']
# y = np.array(class_list, dtype='uint8')
# type(y), y.shape, y[:10]

# u, c = np.unique(y.ravel(), return_counts=True)
# print(u, c, np.sum(c))


#%%
from skimage.io import imread
class_list = np.array(class_list) - 1

trainPathsDf = pd.DataFrame({
    'class': class_list,
    'path': trainPaths
})
print(type(trainPathsDf))

# features['path'] = trainPathsDf[0].map(lambda x: x)
# features.sample(6)

# with tf.device('/device:GPU:0'):
#     features['image'] = features['path'].map(imread)

# img = imread(trainPathsDf['path'].iloc[0])
# plt.imshow(img)
        
#%%
y_series = trainPathsDf['class'] 

if platform.system() != 'Linux':
    # Plotting
    fig = plt.figure(dpi=300)
    palette = sns.color_palette("Set2", 3)
    ax = sns.countplot(x=y_series, palette=palette)
    
    # Sort class_dict by keys and extract labels in sorted order
    # ordered_labels = [class_dict[key] for key in sorted(class_dict)]
    class_labels = config['TRAIN']['CLASS_LABELS'] 
    # Set x-tick labels
    ax.set_xticklabels(class_labels)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xlabel("Class")
    plt.show()

    # fig.savefig('unbalance-class-counts-3TTMs.png')


#%%
X_Shape = X.shape[1]*X.shape[2]*X.shape[3]
X_Flat = X.reshape(X.shape[0], X_Shape)
print(X.shape, X_Flat.shape)

#%
ros = RandomOverSampler()
# X_Ros, y_Ros = ros.fit_sample(X_Flat, y) #py36
X_Ros, y_Ros = ros.fit_resample(X_Flat, y_series)

#X_RosFeature, y_Ros1 = ros.fit_sample(x_feature, y)
#X_RosSymptom, y_Ros2 = ros.fit_sample(X_symptom, y)
print(X_Ros.shape, y_Ros.shape)

#%%
if platform.system() != 'Linux':
    fig = plt.figure(dpi=300)
    # u, c = np.unique(y.ravel(), return_counts=True)
    # print(u, c, np.sum(c))
    palette = sns.color_palette("Set2", 3)
    ax = sns.countplot(x=y_Ros, palette=palette)
    ax.set_xticklabels(class_labels)

    # Add class count labels on each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xlabel("Class")
    plt.show()
    # fig.savefig('balance-class-counts-3TTMs.png')

#%%
for i in range(len(X_Ros)):    
    X_RosReshaped = X_Ros.reshape(len(X_Ros), 
                    config['DATA']['W'],
                    config['DATA']['H'],
                    config['DATA']['C'])
print("X_Ros Shape: ", X_Ros.shape)
print("X_RosReshaped Shape: ", X_RosReshaped.shape)

#%
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_RosHot = to_categorical(y_Ros, 
            num_classes = config['MODEL']['NUM_CLASSES'])

print(y_RosHot.shape)
   
#%%
# Example usage: setting the environment for each seed in _C.DATA.SEEDS
model_preds = list()
 
for seed in config['DATA']['SEEDS']:
    print('\n\nseed: ', seed)
    seed_everything(seed)
    # model_preds = list()
    
    for MODEL in config['MODEL']['NAMES']:
        # MODEL = 'Xception'
        print(MODEL)
        
        if config['EVALUATE'] == True:
            train_log =  config['BASE'] + '-' + \
                    'Eva-' + \
                    config['DATA']['SET'] + '-' + \
                    'seed' + str(seed) + '-' + \
                    config['DATA']['DEVICE'] + '-' + \
                    config['TRAIN']['DATETIME']
        else:
            if config['TRAIN']['TL'] == True:
                train_log = config['BASE'] + '-' + \
                    config['DATA']['SET'] + '-' + \
                    'seed' + str(seed) + '-' + \
                    config['DATA']['DEVICE'] + '-' + \
                    MODEL + '-' + \
                    'TL' + '-' + \
                    config['TRAIN']['DATETIME']
            else:
                train_log = config['BASE'] + '-' + \
                    config['DATA']['SET'] + '-' + \
                    'seed' + str(seed) + '-' + \
                    config['DATA']['DEVICE'] + '-' + \
                    MODEL + '-' + \
                    config['TRAIN']['DATETIME']
        
        print(train_log)
        
        #%%
        if config['SAVE'] == True:
            cfg_dict = cfg_to_dict(config)
            json_file_path = train_log + '_config.json'  
            with open(json_file_path, 'w') as json_file:    
                json.dump(cfg_dict, json_file, indent=4)
            
            print("Saved: ", json_file_path)
                      
        #%%
        # Split the train and the validation set for the fitting
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        
        X_train, X_test, Y_train, Y_test = train_test_split(
                # X_RosReshaped[:SAMPLES],
                # y_RosHot[:SAMPLES],  
                X_RosReshaped,
                y_RosHot, 
                test_size = config['DATA']['TEST_SIZE'],
                random_state=seed
                )
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, 
            test_size = config['DATA']['TEST_SIZE'], 
            random_state=seed)
        
        X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=seed)
        X_val, Y_val = shuffle(X_val, Y_val, random_state=seed)
        
        print("X_train shape", X_train.shape)
        print("Y_train shape", Y_train.shape)
        print("X_test shape", X_test.shape)
        print("Y_test shape", Y_test.shape)
        print("X_val shape", X_val.shape)
        print("Y_val shape", Y_val.shape)
        # print("trainAttrX shape",trainAttrX.shape)
        # print("testAttrX shape",testAttrX.shape)
        
        #%%
        def count_plot(label, title):
            label = np.argmax(label, axis=1)
            fig = plt.figure(dpi=300)
            
            ax = sns.countplot(x=label)
            # Add class count labels on each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')
            plt.title(title)
            plt.show()
            u, c = np.unique(label.ravel(), return_counts=True)
            print(u, c, np.sum(c))
            prefix = config['BASE'] + '-' + config['DATA']['SET'] + '-'
            filename = prefix + title + '.png'
            # fig.savefig(filename)
        
        if platform.system() != 'Linux':
            count_plot(Y_train, 'Train set') 
            count_plot(Y_val, 'Validation set')
            count_plot(Y_test, 'Test set')
        
        #%%
        def data_augment(image):
            p_spatial = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
            p_rotate = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
            p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
            p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
            p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
            
            # Flips
            image = tf.image.random_flip_left_right(image)
            #image = tf.image.random_flip_up_down(image)
            
            if p_spatial > .75:
                image = tf.image.transpose(image)
                
            # Rotates
            # if p_rotate > .75:
            #     image = tf.image.rot90(image, k = 3) # rotate 270º
            # elif p_rotate > .5:
            #     image = tf.image.rot90(image, k = 2) # rotate 180º
            # elif p_rotate > .25:
            #     image = tf.image.rot90(image, k = 1) # rotate 90º
                
            # Pixel-level transforms
            if p_pixel_1 >= .5:
                image = tf.image.random_saturation(image, lower = .2, upper = 1.0)
            if p_pixel_2 >= .5:
                image = tf.image.random_contrast(image, lower = .2, upper = 1.0)
                pass
            if p_pixel_3 >= .3:
                image = tf.image.random_brightness(image, max_delta = .1)
                
            return image
        
        #%%
        # MODEL = 'DenseNet121'  
        MODEL = 'Xception'    
        if MODEL == 'VGG16':
            preprocessing_function = vgg16.preprocess_input
        elif MODEL == 'VGG19':
            preprocessing_function = vgg19.preprocess_input
        elif MODEL == 'EfficientNetB7':
            preprocessing_function = efficientnet.preprocess_input
        elif MODEL == 'Xception':
            preprocessing_function = xception.preprocess_input
        elif MODEL == 'InceptionResNetV2':
            preprocessing_function = inception_resnet_v2.preprocess_input
        elif MODEL == 'InceptionV3':
            preprocessing_function = inception_v3.preprocess_input
        elif MODEL == 'DenseNet121':
            preprocessing_function = densenet.preprocess_input
        elif MODEL == 'MobileNet':
            preprocessing_function = mobilenet.preprocess_input   
        elif MODEL == 'MobileNetV2':
            preprocessing_function = mobilenet_v2.preprocess_input
        else:
            preprocessing_function = vit.preprocess_inputs
            
        #%%        
        # if "ViT" in MODEL.split('_'):
        #     model = create_vit_model(MODEL)           
        # else:
        #     model = create_cnn_model(MODEL)
        
        # print('Created: ', model.name)
        
      
        #%%    
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale = 1./255,
            # samplewise_center = True,
            # samplewise_std_normalization = True,
            # preprocessing_function = data_augment,
            # rotation_range=10,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rescale=1./255,
            # shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            # fill_mode='reflect',
            preprocessing_function = preprocessing_function,
            )
        # set as training data
        datagen.fit(X_train)
        
        train_gen  = datagen.flow(
            X_train, Y_train,
            #target_size=(224, 224),
            batch_size = config['TRAIN']['BATCH_SIZE'],
            seed = seed,
            # color_mode = 'rgb',
            shuffle = True,
            # class_mode='categorical',
            # subset='training'
            ) 
        
        # same directory as training data
        # val_datagen = ImageDataGenerator(
        #     rescale=1./255
        #     )
        
        valid_gen  = datagen.flow(
            X_val, Y_val,
            #target_size=(224, 224),
            batch_size = config['TRAIN']['BATCH_SIZE'],
            seed = seed,
            # color_mode = 'rgb',
            shuffle = True,
            # class_mode='categorical',
            # subset='training'
            ) 
        
        test_gen = datagen.flow(
                X_test, Y_test,
                #target_size=(224, 224),
                batch_size = config['TRAIN']['BATCH_SIZE'],
                seed = seed,
                # color_mode = 'rgb',
                shuffle = False,
                # class_mode='categorical',
                # subset='training'
                ) 
        
        
        #%%
        if platform.system() != 'Linux':
            images, labels = train_gen.next()
            # images, labels = valid_gen.next()
            print(images.shape, labels.shape)
            
            # images = [train_gen[0][0][i] for i in range(16)]
            fig, axes = plt.subplots(2, 4, figsize = (8, 4), dpi=300)
            
            axes = axes.flatten()
            
            for img, ax in zip(images, axes):
                # ax.imshow(
                #     img.reshape(config['DATA']['W'],
                #                 config['DATA']['H'],
                #                 config['DATA']['C']).astype("uint8"),
                #     # cmap='gray'
                #     )
                
                ax.imshow(img, 
                          # cmap='gray'
                          )              
                ax.axis('on')
            
            plt.tight_layout()
            plt.show()
            # fig.savefig('aug-samples-D1-DSLR-densenet.png', dpi=300)
            fig.savefig('aug-samples-D1-DSLR-exception.png', dpi=300)
            # plt.imshow(image.squeeze(), cmap='gray')
            # plt.axis('off')
            # plt.show()
        
        #%%        
        from tensorflow.keras.models import load_model
        # backup_model_best = 'E:/Tongue/models/Windows-B-seed2024-Mobile-InceptionV3-20240423-1744.hdf5'
        if MODEL == 'DenseNet121':
            backup_model_best = "E:/Tongue/model-best/Windows-A-seed1337-Mobile-DenseNet121-TL-20240403-1626.hdf5"
        else:
            backup_model_best = "E:/Tongue/model-best/Windows-A-seed42-Mobile-Xception-TL-20240425-1525.hdf5"
        print(backup_model_best)
        best_model = load_model(backup_model_best)
        # best_model.summary()
        print("Loaded: ", backup_model_best)
        
        #%%
        # print("Evaluate Val set")
        # scores = best_model.evaluate(valid_gen, verbose=1)
        
        # for metric, value in zip(best_model.metrics_names, scores):
        #     print("mean {}: {:.2}".format(metric, value))
            
        # print() 
        
        # #%
        # print("Evaluate Test set")
        # scores_test = best_model.evaluate(test_gen, verbose=1)
        # for metric, value in zip(best_model.metrics_names, scores_test):
        #     print("mean {}: {:.2}".format(metric, value))
            
        # print() 
        
        #%%
        from datetime import datetime
        print("\nEvaluate the whole test set")
        t3 = datetime.now()
        y_test_pred_list = []
        pair_idx_test_list = []
           
        for batch in tqdm(range(test_gen.__len__())):
            # print('\nPredicting batch: ', batch)
            X_test, y_test = test_gen.__getitem__(batch)
            print(X_test.shape, y_test.shape) 
            # print(X_test.shape, y_test.shape, len(pair_idx_test))
            # y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')
            verbose=2
            with tf.device('/device:GPU:0'):
                y_test_pred = best_model.predict(
                    # test_generator_RBF.__getitem__(image_number),
                    X_test,   
                    batch_size=config['TRAIN']['BATCH_SIZE'], 
                    verbose=verbose)
        
            y_test_pred_list.append(y_test_pred)
              
        y_test_pred = np.concatenate(y_test_pred_list, axis=0)
        print(y_test_pred.shape)
        model_preds.append(y_test_pred)
        
        y_test_pred_argmax = np.argmax(y_test_pred, axis=1)
        print(y_test_pred_argmax.shape)
        
        t4 = datetime.now() - t3
        print('Execution times: ', t4, '\n')
        
        # del best_model
  
#%%
# stop

preds = np.array(model_preds)
print(preds.shape)  
preds_sum =   preds.sum(axis=0)  
print(preds_sum.shape)  
preds_sum_argmax = np.argmax(preds_sum, axis=1)
print(preds_sum_argmax.shape)

#%%
true_classes = np.argmax(Y_test, axis=1)
# predicted_classes = y_test_pred_argmax
predicted_classes = preds_sum_argmax
# class_labels = list(valid_gen.class_indices.keys())  
print('Test set')
print(classification_report(true_classes,
                predicted_classes,
                target_names=config['TRAIN']['CLASS_LABELS'])
      )

class_report = classification_report(true_classes,
                predicted_classes,
                target_names=config['TRAIN']['CLASS_LABELS'],
                output_dict = True)

#%%
confusionmatrix = confusion_matrix(true_classes,
                            predicted_classes)

FP = confusionmatrix.sum(axis=0) - np.diag(confusionmatrix)  
FN = confusionmatrix.sum(axis=1) - np.diag(confusionmatrix)
TP = np.diag(confusionmatrix)
# TN = confusionmatrix.values.sum() - (FP + FN + TP)
TN = confusionmatrix.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
sen = TPR.mean()
spec = TNR.mean()

print(sen, spec)

p_measures = dict()
p_measures = {
    # 'Training': t2,
    
    'TP': TP,
    'TN': TN,
    'FP': FP,
    'FN': FN,
    'TPR': TPR,
    'TNR': TNR,
    'PPV': PPV,
    'NPV': NPV,
    'FPR': FPR,
    'FNR': FNR,
    'FDR': FDR,
    'ACC': ACC,
    
    'prescision': class_report['macro avg']['precision'],
    'f1-score' : class_report['macro avg']['f1-score'],
    'MeanAcc': ACC.mean(),
    'MeanSen': sen,
    'MeanSpec': spec,           
}


# Convert numpy arrays to lists
for key, value in p_measures.items():
    if isinstance(value, np.ndarray):
        p_measures[key] = value.tolist()
print(p_measures)
       
# Define file path
file_path = train_log + '_p_measures.json'

# Write to JSON file
with open(file_path, 'w') as file:
    json.dump(p_measures, file, indent=4)

print(file_path)

#%%
fig = plt.figure(figsize = (6, 4), dpi=300)
sns.heatmap(confusionmatrix,
            cmap = 'Blues',
            annot = True,
            cbar = True,
            xticklabels=config['TRAIN']['CLASS_LABELS'],
            yticklabels=config['TRAIN']['CLASS_LABELS'])
plt.ylabel('Actual', fontsize=10)
plt.xlabel("Predicted\n \
           Accuracy={:0.2f}\n \
           Sensitivity={:0.2f}\n \
           Specificity={:0.2f}\n".format(ACC.mean(),
                                   TPR.mean(),
                                   TNR.mean())
           ,
          fontsize=10)
plt.tight_layout()
plt.show()

filename = train_log + '-confuseMatrix.png'
fig.savefig(filename)
print(filename)

#%%
confusionmatrix_normalized = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize = (6, 4), dpi=300)
sns.heatmap(confusionmatrix_normalized,
            cmap = 'Blues',
            annot = True,
            cbar = True,
            fmt=".2f",
            xticklabels=config['TRAIN']['CLASS_LABELS'],
            yticklabels=config['TRAIN']['CLASS_LABELS']
            )
plt.ylabel('Actual', fontsize=10)
plt.xlabel("Predicted\n \
           Accuracy={:0.2f}\n \
           Sensitivity={:0.2f}\n \
           Specificity={:0.2f}\n".format(ACC.mean(),
                                   TPR.mean(),
                                   TNR.mean())
           ,
          fontsize=10)
plt.tight_layout()
plt.show()
filename = train_log + '-confuseMatrixNorm.png'
fig.savefig(filename)
print(filename)

#%%

