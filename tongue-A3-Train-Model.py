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
import configs
config = configs.get_config()

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

def seed_everything(seed = config['DATA']['SEED']):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()

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

train_log = config['BASE'] + '-' + \
    config['DATA']['SET'] + '-' + \
    config['DATA']['DEVICE'] + '-' + \
    config['MODEL']['NAME'] + '-' + \
    config['TRAIN']['DATETIME']
print(train_log)


# cfg_dict = cfg_to_dict(config)
# json_file_path = train_log + '_config.json'  
# with open(json_file_path, 'w') as json_file:    
#     json.dump(cfg_dict, json_file, indent=4)

# print("Saved: ", json_file_path)

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
                             
    print(case_id, specific_class)
    class_list.append(int(specific_class))

print('len: ', len(class_list))

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
# x_feature = features[['Shape', 
#                       'Color',
#                       'Moisture', 
#                       'Coating', 
#                       'Midline',
#                       'Crack', 
#                       'Teethmarks',
#                       #'Class'
#                       ]]

# x_feature.sample(5)


# #%%
# corrMatrix = x_feature.corr()
# plt.figure(figsize=(8,6), dpi=300)
# sns.heatmap(corrMatrix, annot=True)
# plt.show()

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
if platform.system() != 'Linux':
    n_samples = 5
    num_classes = 3
    fig, m_axs = plt.subplots(num_classes, 
            n_samples,
            figsize = (num_classes*n_samples, num_classes*n_samples),
            dpi=120)
    for n_axs, (type_name, type_rows) in zip(m_axs, 
            trainPathsDf.sort_values(['class']).groupby('class')):
        #n_axs[type_name].set_title(class_dict[type_name])
        for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2020).iterrows()):
            #print(c_row)
            #n_axs[type_name].set_title(class_dict[type_name])
            img = imread(c_row['path'])
            c_ax.set_title(class_dict[type_name])
            c_ax.imshow(img)
            c_ax.axis('on')
    plt.tight_layout()
    # fig.savefig('category-samples-A-Mobile-PJ.png')


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

    # fig.savefig('unbalance-class-counts-PJ.png')


#%%
X_Shape = X.shape[1]*X.shape[2]*X.shape[3]
X_Flat = X.reshape(X.shape[0], X_Shape)
print(X.shape, X_Flat.shape)

#%%
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
    
    # Add class count labels on each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xlabel("Class")
    plt.show()
    # fig.savefig('balance-class-counts-PJ.png')

#%%
for i in range(len(X_Ros)):    
    X_RosReshaped = X_Ros.reshape(len(X_Ros), 
                    config['DATA']['W'],
                    config['DATA']['H'],
                    config['DATA']['C'])
print("X_Ros Shape: ", X_Ros.shape)
print("X_RosReshaped Shape: ", X_RosReshaped.shape)

#%%
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_RosHot = to_categorical(y_Ros, 
            num_classes = config['MODEL']['NUM_CLASSES'])

print(y_RosHot.shape)

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
        random_state=config['DATA']['SEED']
        )

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, 
    test_size = config['DATA']['TEST_SIZE'], 
    random_state=config['DATA']['SEED'])

X_train, Y_train = shuffle(X_train, Y_train, random_state=config['DATA']['SEED'])
X_test, Y_test = shuffle(X_test, Y_test, random_state=config['DATA']['SEED'])
X_val, Y_val = shuffle(X_val, Y_val, random_state=config['DATA']['SEED'])

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
# from tensorflow.keras.applications.efficientnet import (
#     EfficientNetB7, preprocess_input)
from tensorflow.keras.applications import (vgg16, vgg19,
    EfficientNetB7, xception, efficientnet, inception_resnet_v2,
    inception_v3, densenet, resnet, resnet_v2, 
    mobilenet, mobilenet_v2)

if config['MODEL']['NAME'] == "ViT":
    preprocessing_function = vit.preprocess_inputs

def create_cnn_model(model_name):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])   
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = vgg19.preprocess_input
    
    elif model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = efficientnet.preprocess_input
    
    elif model_name == 'Xception':
        base_model = Xception(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = xception.preprocess_input
    
    elif model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = inception_resnet_v2.preprocess_input
    
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = inception_v3.preprocess_input
    
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = densenet.preprocess_input
    
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = mobilenet.preprocess_input
        
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet',
                include_top=False,
                input_shape = config['DATA']['DIMENSION'])
        # preprocessing_function = mobilenet_v2.preprocess_input
    
    
    #customLayers = GlobalAveragePooling2D()(base_model.output)
    customLayers = Flatten()(base_model.output)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(1024,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(512,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)
    
    customLayers=Dense(256,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(64,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(32,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(16,activation='relu')(customLayers)
    customLayers=Dropout(0.2)(customLayers)

    customLayers=Dense(config['MODEL']['NUM_CLASSES'],
            activation='softmax')(customLayers)


    model = Model(base_model.input, customLayers, name=model_name)
    
    base_model.trainable = False
    return model

def create_vit_model(model_name):
    if model_name == 'ViT_b16':
        base_model = vit.vit_b16(
          image_size = config['DATA']['IMAGE_SIZE'],
            # activation = 'softmax',
            activation = tfa.activations.gelu,
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = config['MODEL']['NUM_CLASSES'])
        
    elif model_name == 'ViT_l16':
        base_model = vit.vit_l16(
            image_size = config['DATA']['IMAGE_SIZE'],
            # activation = 'softmax',
            activation = tfa.activations.gelu,
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = config['MODEL']['NUM_CLASSES'])
    
    elif model_name == 'ViT_b32':
        base_model = vit.vit_b32(
            image_size = config['DATA']['IMAGE_SIZE'],
            # activation = 'softmax',
            activation = tfa.activations.gelu,
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = config['MODEL']['NUM_CLASSES'])
        
    else:
        base_model = vit.vit_l32(
            image_size = config['DATA']['IMAGE_SIZE'],
            # activation = 'softmax',
            activation = tfa.activations.gelu,
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = config['MODEL']['NUM_CLASSES'])
       
    #customLayers = GlobalAveragePooling2D()(base_model.output)
    customLayers = Flatten()(base_model.output)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(1024,
                       # activation='relu',
                       activation = tfa.activations.gelu
                       )(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(512,
                       # activation='relu',
                       activation = tfa.activations.gelu
                       )(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(64,
                       # activation1='relu',
                       activation = tfa.activations.gelu
                       )(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(32,
                       # activation='relu',
                       activation = tfa.activations.gelu
                       )(customLayers)
    customLayers=Dropout(0.2)(customLayers)
    customLayers = BatchNormalization()(customLayers)

    customLayers=Dense(16, 
                       # activation='relu',
                       activation = tfa.activations.gelu
                       )(customLayers)
    customLayers=Dropout(0.2)(customLayers)

    customLayers=Dense(config['MODEL']['NUM_CLASSES'],
            activation='softmax')(customLayers)


    model = Model(base_model.input, customLayers, name=model_name)
    
    base_model.trainable = False
    return model

#%%
if config['MODEL']['NAME'] == "ViT":
    preprocessing_function = vit.preprocess_inputs
elif config['MODEL']['NAME'] == 'VGG16':
    preprocessing_function = vgg16.preprocess_input
elif config['MODEL']['NAME'] == 'VGG19':
    preprocessing_function = vgg19.preprocess_input
elif config['MODEL']['NAME'] == 'EfficientNetB7':
    preprocessing_function = efficientnet.preprocess_input
elif config['MODEL']['NAME'] == 'Xception':
    preprocessing_function = xception.preprocess_input
elif config['MODEL']['NAME'] == 'InceptionResNetV2':
    preprocessing_function = inception_resnet_v2.preprocess_input
elif config['MODEL']['NAME'] == 'InceptionV3':
    preprocessing_function = inception_v3.preprocess_input
elif config['MODEL']['NAME'] == 'DenseNet121':
    preprocessing_function = densenet.preprocess_input
elif config['MODEL']['NAME'] == 'MobileNet':
    preprocessing_function = mobilenet.preprocess_input   
elif config['MODEL']['NAME'] == 'MobileNetV2':
    preprocessing_function = mobilenet_v2.preprocess_input
    
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
    # preprocessing_function = preprocessing_function,
    )
# set as training data
datagen.fit(X_train)

train_gen  = datagen.flow(
    X_train, Y_train,
    #target_size=(224, 224),
    batch_size = config['TRAIN']['BATCH_SIZE'],
    seed = config['DATA']['SEED'],
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
    seed = config['DATA']['SEED'],
    # color_mode = 'rgb',
    shuffle = True,
    # class_mode='categorical',
    # subset='training'
    ) 

test_gen = datagen.flow(
        X_test, Y_test,
        #target_size=(224, 224),
        batch_size = config['TRAIN']['BATCH_SIZE'],
        seed = config['DATA']['SEED'],
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
        ax.imshow(
            img.reshape(config['DATA']['W'],
                        config['DATA']['H'],
                        config['DATA']['C']).astype("uint8"),
            # cmap='gray'
            )
        ax.axis('on')
    
    plt.tight_layout()
    plt.show()
    # fig.savefig('aug-samples-mobile.png', dpi=300)
    
    # plt.imshow(image.squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.show()

#%%
# from vit_keras import vit, utils

# classes = utils.get_imagenet_classes()
# model = vit.vit_b16(
#     image_size=384,
#     activation='sigmoid',
#     pretrained=True,
#     include_top=True,
#     pretrained_top=True
# )
# url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg'
# image = utils.read(url, image_size)
# X = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
# y = model.predict(X)
# print(classes[y[0].argmax()]) # Granny smith

#%%

# import numpy as np
# import matplotlib.pyplot as plt
# from vit_keras import vit, utils, visualize

# # Load a model
# classes = utils.get_imagenet_classes()
# model = vit.vit_b16(
#     image_size=image_size,
#     activation='sigmoid',
#     pretrained=True,
#     include_top=True,
#     pretrained_top=True
# )
# classes = utils.get_imagenet_classes()

# # Get an image and compute the attention map
# url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
# # image = utils.read(url, image_size)
# image = X_train[99]
# attention_map = visualize.attention_map(model=model, image=image)
# print('Prediction:', classes[
#     model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
# )  # Prediction: Eskimo dog, husky

# # Plot results
# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.axis('off')
# ax2.axis('off')
# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# _ = ax1.imshow(image)
# _ = ax2.imshow(attention_map)


#%%
# from vit_keras import vit

# vit_model = vit.vit_b16(
#         image_size = image_size,
#         activation = 'softmax',
#         pretrained = True,
#         include_top = False,
#         pretrained_top = False,
#         classes = 3)

# vit_model.summary()

#%%
class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#%%
# plt.figure(figsize=(4, 4))
# batch_size = 16
# patch_size = 7  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2

# # x = train_gen.next()
# # image = x[0][0]

# x = X_train[0]
# image = x

# plt.imshow(image.astype('uint8'))
# plt.axis('off')

# resized_image = tf.image.resize(
#     tf.convert_to_tensor([image]), size = (image_size, image_size)
# )

# print(resized_image.shape)

# patches = Patches(patch_size)(resized_image)
# print(f'Image size: {image_size} X {image_size}')
# print(f'Patch size: {patch_size} X {patch_size}')
# print(f'Patches per image: {patches.shape[1]}')
# print(f'Elements per patch: {patches.shape[-1]}')

# n = int(np.sqrt(patches.shape[1]))
# plt.figure(figsize=(4, 4))

# for i, patch in enumerate(patches[0]):
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#     plt.imshow(patch_img.numpy().astype('uint8'))
#     plt.axis('off')
    
# #%%
# model = tf.keras.Sequential([
#         vit_model,
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
#         tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
#         tf.keras.layers.Dense(3, 'softmax')
#     ],
#     name = 'vision_transformer')

# model.summary()


#%%
# warnings.filterwarnings("ignore")

# optimizer = tfa.optimizers.RectifiedAdam(
#     learning_rate = learning_rate)

# model.compile(optimizer = optimizer, 
#   loss = tf.keras.losses.CategoricalCrossentropy(
#       label_smoothing = 0.2), 
#               metrics = ['accuracy'])

# STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
# STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

# # STEP_SIZE_TRAIN = len(X_train) // batch_size

# backup_model_best = 'ViT-model-WS-2023-12-24.hdf5'
# print('\nbackup_model_best: ', backup_model_best)

# mcp2 = ModelCheckpoint(backup_model_best,
#                 monitor='val_loss',
#                 verbose=1,
#                 save_best_only=True,
#                 mode='min'
#                 )  

# early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
#     patience = 15,
#     restore_best_weights = True,
#     verbose = 1)

#%%
# X_train_rescaled = X_train/255.0

#%%
# history = model.fit(
#     train_gen,  
#     # steps_per_epoch = STEP_SIZE_TRAIN,
#     validation_data = valid_gen,
#     # validation_steps = STEP_SIZE_VALID,
#     epochs = EPOCHS,
#     verbose=1,
#     callbacks = [
#         mcp2,
#         early_stopping_callbacks,
#         tqdm_callback, 
#         PlotLossesKeras(),
#         ]
#           )

# #%%
# from tensorflow.keras.models import Model, load_model
# model = load_model(backup_model_best)

# #%
# predicted_classes = np.argmax(
#     model.predict(X_test, 
#                   verbose=1,
#     steps = len(X_test) // batch_size + 1),
#     axis = 1)

#%
# true_classes = valid_gen.classes
# true_classes = np.argmax(Y_test, axis=1)
# class_labels = list(valid_gen.class_indices.keys())  

# confusionmatrix = confusion_matrix(true_classes, predicted_classes)
# plt.figure(figsize = (8, 8), dpi=300)
# sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

# print(classification_report(true_classes, predicted_classes))


#%%
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class LRTensorBoard(Callback):
    def __init__(self, log_dir, init_lr, lr_multiplier, file):
        super().__init__()
        self.log_dir = log_dir
        self.init_lr = init_lr
        self.lr_multiplier = lr_multiplier
        self.iterations = 0
        self.losses = []
        self.lrs = []
        self.file = file
    
    def on_train_batch_end(self, batch, logs=None):
        lr = self.init_lr * (self.lr_multiplier ** self.iterations)
        self.lrs.append(lr)
        self.losses.append(logs['loss'])
        self.iterations += 1
    
    def on_epoch_end(self, epoch, logs=None):
        if self.iterations == 0:
            return
        plt.figure(figsize=(8, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True)
        plt.savefig(f'{self.log_dir}/{self.file}')
        plt.close()



#%%
# from tensorflow.keras import optimizers
# # Set LR range test parameters
# initial_lr = 1e-5 # Initial LR
# desired_lr = 1
# # num_steps = train_steps_RBF//(batch_size*2)
# num_steps = EPOCHS
# plotfile = 'lr_range_test_plot_vgg16_06.png'
# lr_multiplier = (desired_lr / initial_lr) ** (1 / num_steps)

# print(batch_size, num_steps)
# print(initial_lr, desired_lr)
# print(num_steps)
# print(lr_multiplier)


# # Create a LR range test callback
# lr_range_test = LRTensorBoard(
#     log_dir='.',
#     init_lr=initial_lr, 
#     lr_multiplier=lr_multiplier,
#     file=plotfile)
# optimizer = optimizers.Adam(learning_rate = initial_lr) #01

# # optimizer = tfa.optimizers.RectifiedAdam( #02
# #     learning_rate = initial_lr)

# #03
# optimizer = optimizers.SGD(learning_rate = initial_lr)

# # model.compile(optimizer = optimizer, 
# #     loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
# #             metrics = ['accuracy'])

# vgg16_model = create_cnn_model()

# vgg16_model.compile(optimizer=optimizer, 
#             loss='categorical_crossentropy', 
#             metrics=['accuracy']
#             )

# #%%
# vgg16_model.reset_states()
# history = vgg16_model.fit(
#     train_gen,  
#     # steps_per_epoch = STEP_SIZE_TRAIN,
#     validation_data = valid_gen,
#     # validation_steps = STEP_SIZE_VALID,
#     epochs = 1,
#     verbose=1,
#     callbacks = [
#         # mcp2,
#         # early_stopping_callbacks,
#         # tqdm_callback, 
#         # PlotLossesKeras(),
#         lr_range_test,
#         ]
#         )

#%%
# if config['MODEL']['NAME'] == "ViT":
#     model_name = config['MODEL']['NAME'] + config['MODEL']['SUB_NAME']
#     # model_name = 'ViT_b16'
#     model = create_vit_model(model_name)
# else:
#     model = create_cnn_model(config['MODEL']['NAME'])

# print(model.summary())


#%%
from tensorflow.keras.models import load_model
print("\nEvaluate model for the val set")
backup_model_best = os.path.join(config['BASEPATH'],
    'models',
    'Windows-C-DSLR-ARUN-20240401-2009.hdf5')
print('\nbackup_model_best: ', backup_model_best)
best_model = load_model(backup_model_best, compile=False)
# print(best_model.summary())
print('\nLoaded: ' , backup_model_best)

#%% Create a model that will return the outputs of the selected_layer
selected_layer_name = 'conv2d_155'
print(selected_layer_name)
selected_layer = best_model.get_layer(selected_layer_name)
base_model = tf.keras.models.Model(
    inputs = best_model.input,
    outputs = selected_layer.output)
  
# base_model.summary()

# base_model.trainable = False
base_model.trainable = True
# customLayers = GlobalAveragePooling2D()(base_model.output)
customLayers = Flatten()(base_model.output)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(1024,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(512,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(256,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(64,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(32,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)
customLayers = BatchNormalization()(customLayers)

customLayers=Dense(16,activation='relu')(customLayers)
customLayers=Dropout(0.2)(customLayers)

customLayers=Dense(config['MODEL']['NUM_CLASSES'],
        activation='softmax')(customLayers)

model_name = config['MODEL']['NAME']
tl_model = tf.keras.models.Model(base_model.input,
                                 customLayers,
                                 name=model_name)

print(tl_model.summary())

#%%
# optimizer_adam = optimizers.Adam(
#     learning_rate = config['TRAIN']['LR'])

# optimizer_radam = tfa.optimizers.RectifiedAdam(
#     learning_rate = config['TRAIN']['LR'])

# optimizer_adamw = tfa.optimizers.AdamW(
#     learning_rate=config['TRAIN']['LR'],
#     weight_decay=config['TRAIN']['LR'])

# loss = tf.keras.losses.CategoricalCrossentropy()
# loss_smooth = tf.keras.losses.CategoricalCrossentropy(
#     label_smoothing = 0.2)

tl_model.compile(
    optimizer='adam',
    # optimizer='sgd',
    # optimizer=optimizer_adam,
    # optimizer=tfa.optimizers.RectifiedAdam(),
    loss='categorical_crossentropy',
    # loss=loss_smooth,
    metrics=['accuracy']
    )

backup_model_best = os.path.join(
    config['DATA']['SAVEPATH'],
    f'{train_log}.hdf5')

print('\nbackup_model_best: ', backup_model_best)

mcp2 = ModelCheckpoint(backup_model_best,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min'
                )

reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.8,
    # factor=0.9,
    patience=3, 
    verbose=1,
    mode='auto', 
    # min_delta=1e-3,
    # cooldown=4,
    # min_lr=1e-3
    )

early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
    patience = 10,
    restore_best_weights = True,
    verbose = 1)


#%%
from tensorflow.keras.models import load_model
t1 = time.time()

# model.reset_states()
if config['TRAIN']['TL'] == True:
    print('\nTL training...\n')
    # load trained weights of A-DSLR
    # init_file = 'Windows-A-DSLR-ViT-20240325-1832.hdf5'
    # init_file = 'Windows-A-DSLR-VGG16-20240325-1716.hdf5'
    init_backup_model_best = os.path.join(config['DATA']['SAVEPATH'],
                            config['MODEL']['BEST'])
    best_model = load_model(init_backup_model_best)
    # best_model.summary()
    print("Loaded: ", init_backup_model_best)
    # train A-Mobile
    history2 = best_model.fit(
        train_gen,  
        # steps_per_epoch = STEP_SIZE_TRAIN,
        validation_data = valid_gen,
        # validation_steps = STEP_SIZE_VALID,
        epochs = config['TRAIN']['EPOCHS'],
        # epochs = 3,
        verbose=2,
        callbacks = [
            mcp2,
            reduceLROnPlat,
            early_stopping_callbacks,
            tqdm_callback, 
            PlotLossesKeras(),
            ]
          )
    
else:
    with tf.device('/device:GPU:0'):
        history2 = tl_model.fit(
            train_gen,  
            # steps_per_epoch = STEP_SIZE_TRAIN,
            validation_data = valid_gen,
            # validation_steps = STEP_SIZE_VALID,
            epochs = config['TRAIN']['EPOCHS'],
            # epochs = 3,
            verbose=2,
            callbacks = [
                mcp2,
                reduceLROnPlat,
                early_stopping_callbacks,
                tqdm_callback, 
                PlotLossesKeras(),
                ]
              )
        
t2 = (time.time()-t1)/60
print('\nTime used: {:2.2f} min'.format( t2 ))

#%%
model_history_df = pd.DataFrame(history2.history) 

# with open('unet_history_df.csv', mode='w') as f:
#     unet_history_df.to_csv(f)
    
# with open('att_unet_history_df.csv', mode='w') as f:
#     att_unet_history_df.to_csv(f)

history_file = f'{train_log}_history_df.csv'
# history_file_path = os.path.join(model_name, history_file)
with open(history_file, mode='w') as f:
    model_history_df.to_csv(f)  
print("\nSaved: ", history_file)

#%%
# if LOGSCREEN==True:

history = history2

#%plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and validation loss ' + train_log)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss_file = f'{train_log}-loss.png'
# loss_file_path = os.path.join(model_name, loss_file)
# fig.savefig(loss_file)

#%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy ' + train_log)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
dice_file = f'{train_log}-accuracy.png'
# dice_file_path = os.path.join(model_name, dice_file)
# fig.savefig(dice_file)


lr =  history.history['lr']
epochs_plot = range(1, len(loss) + 1)
fig = plt.figure(figsize=(8, 6), dpi=600)
plt.plot(epochs_plot, lr, 'b', label='Training LR')
plt.title('Trainining learning rate ' + train_log)
plt.xlabel('Epochs')
plt.ylabel('LR')
plt.legend()
plt.show()
lr_file = f'{train_log}-LR.png'
# lr_file_path = os.path.join(model_name, lr_file)
# fig.savefig(lr_file)

#%%
from tensorflow.keras.models import load_model
# backup_model_best = 'Windows-B-ViT-20240320-1203.hdf5'
best_model = load_model(backup_model_best)
# best_model.summary()
print("Loaded: ", backup_model_best)

#%%
print("Evaluate Val set")
scores = best_model.evaluate(valid_gen, verbose=1)

for metric, value in zip(best_model.metrics_names, scores):
    print("mean {}: {:.2}".format(metric, value))
    
print() 

#%
print("Evaluate Test set")
scores_test = best_model.evaluate(test_gen, verbose=1)
for metric, value in zip(best_model.metrics_names, scores_test):
    print("mean {}: {:.2}".format(metric, value))
    
print() 

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

y_test_pred_argmax = np.argmax(y_test_pred, axis=1)
print(y_test_pred_argmax.shape)

t4 = datetime.now() - t3
print('Execution times: ', t4, '\n')

#%%
# predicted_classes = np.argmax(
#     best_model.predict(
#         X_val, 
#         verbose=1,
#     steps = len(X_test) // batch_size + 1),
#     axis = 1)

# #%
# # true_classes = valid_gen.classes
# true_classes = np.argmax(Y_val, axis=1)
# # class_labels = list(valid_gen.class_indices.keys())  
# print(classification_report(true_classes, predicted_classes))


#%%
# predicted_classes = np.argmax(
#     best_model.predict(X_test, 
#                   verbose=1,
#     steps = len(X_test) // batch_size + 1),
#     axis = 1)

#%
true_classes = np.argmax(Y_test, axis=1)
predicted_classes = y_test_pred_argmax
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
    'MeanSpec': spec
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
# fig.savefig(filename)
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
# fig.savefig(filename)
print(filename)

#%%

