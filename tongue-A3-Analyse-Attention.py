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
from tensorflow.keras.utils import to_categorical, Sequence
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

#from tensorflow.keras.preprocessing import image
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

from skimage.io import imread

import mymodels
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

def sanity_check(pairs, batch_idx):   
    for i in batch_idx:
        # image_number = random.randint(0, len(X)-1)
        case_id = os.path.basename(pairs[i][0]).split('-')[0]
        img = imread(pairs[i][0])
        mask = imread(pairs[i][1])
        plt.figure(figsize=(12, 6), dpi=300)
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Image: ' + str(case_id))
        plt.subplot(122)
        plt.imshow(mask, cmap='gray')
        # (unique, counts) = np.unique(y[i], return_counts=True)
        # xlabel = str(unique) + "\n" + str(counts)
        # plt.xlabel(xlabel)
        plt.title('Mask: ' + str(case_id))
        plt.show()
        # print(np.unique(y[i], return_counts=True))
        
        # pair = trainPairs[0]
        # img = imread(pair[0])
        # plt.imshow(img)
#% sanity check
def sanity_check_arr(X, y, note, batch_size=16):   
    for i in range(batch_size):
        # image_number = random.randint(0, len(X)-1)
        plt.figure(figsize=(12, 6), dpi=300)
        plt.subplot(121)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + str(i))
        plt.subplot(122)
        plt.imshow(y[i], cmap='gray')
        # (unique, counts) = np.unique(y[i], return_counts=True)
        # xlabel = str(unique) + "\n" + str(counts)
        # plt.xlabel(xlabel)
        plt.title('Mask: ' + str(i))
        plt.show()
        # print(np.unique(y[i], return_counts=True))
        
class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self,
                 pair, 
                 num_classes=None,
                 batch_size=32, 
                 dim=(256, 256, 3), 
                 shuffle=True,
                 augmentation=None, 
                 preprocessing=None,
                 inference=False,
                 rescale=None,
                 ):
        'Initialization'
        self.dim = dim
        self.pair = pair
        # self.dataset_directory = dataset_directory
        self.num_classes = num_classes
        # self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.inference = inference
        self.rescale = rescale

        # print(self.dataset_directory)
        # print(self.pair)
        print('num_classes: ', self.num_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))
    
    def foo(self):
        print('foo')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.inference:
            # print(list_IDs_temp)
            return X, y, list_IDs_temp
        else:
            return X, y
     
    # def __scale_img(self, single_patch_img):
    #     single_patch_img_scaled = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
    #     return single_patch_img_scaled
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()
        
        # Generate data
        for i in list_IDs_temp:
            # Store sample
            # print(self.pair[i])
            # print(self.pair[i])
            
            # processing image
            # image_path = os.path.join(self.dataset_directory, self.pair[i][0])
            # print(image_path)
            image = load_img(self.pair[i][0], target_size=self.dim)
            image = img_to_array(image, dtype='uint8')
                       
            # processing mask
            dsize = (self.dim[0], self.dim[1])
            # print(dsize)
            # mask = cv2.imread(self.pair[i][1].as_posix(), 0)
            
            # mask_path = os.path.join(self.dataset_directory, self.pair[i][1])        
            mask = cv2.imread(self.pair[i][1], 0)
            mask = cv2.resize(mask, dsize, interpolation = cv2.INTER_NEAREST)
            
             # apply augmentation                     
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']  

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            elif self.rescale==True:
                image = image * (1/255.0)
            elif self.rescale==None:
                # print("Non-preprocess")
                pass
            else:
                # image = self.__scale_img(image)
                pass
            
            masked = mask/255                         
            masked_hot = to_categorical(masked, self.num_classes)
            # print(masked_hot.shape)  
            
            batch_imgs.append(image)
            batch_labels.append(masked_hot)
        
        # print(len(batch_imgs), len(batch_labels))
                       
        return np.array(batch_imgs), np.array(batch_labels)
    

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
    
    batch_idx = random.sample(range(0, len(trainPairs)),
                            config['TRAIN']['BATCH_SIZE'])
    
    # sanity_check(trainPairs, batch_idx)
    
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
# Assuming trainPairs is your list and already defined
# For reproducibility, you might want to shuffle the list first
random.shuffle(trainPairs)

# Calculate split index
split_index = int(len(trainPairs) * 0.7)

# Split trainPairs into two new lists
train_list = trainPairs[:split_index]
val_list = trainPairs[split_index:]

print(f"Train list length: {len(train_list)}")
print(f"Val list length: {len(val_list)}")

#%%
train_generator = DataGenerator(
    train_list,
    num_classes=2,
    batch_size=config['TRAIN']['BATCH_SIZE'], 
    dim=config['DATA']['DIMENSION'],
    # shuffle=True,
    # augmentation=None,
    # inference=False,
    # preprocessing=get_preprocessing(preprocess_input),
    )
train_steps = train_generator.__len__()
print('train_steps_RBF: ', train_steps)

#
image_number = random.randint(0, train_steps)
print('random image number: ', image_number)
X_train, y_train = train_generator.__getitem__(image_number)
print(X_train.shape, y_train.shape)
y_train_argmax = np.argmax(y_train, axis=3).astype('uint8')

#%%
sanity_check_arr(X_train, y_train_argmax,
                note='Train ', 
                batch_size=config['TRAIN']['BATCH_SIZE'])

#%%
val_generator = DataGenerator(
    val_list,
    num_classes=2,
    batch_size=config['TRAIN']['BATCH_SIZE'], 
    dim=config['DATA']['DIMENSION'],
    # shuffle=True,
    # augmentation=None,
    # inference=False,
    # preprocessing=get_preprocessing(preprocess_input),
    )
val_steps = val_generator.__len__()
print('val_steps_RBF: ', val_steps)

#
image_number = random.randint(0, val_steps)
print('random image number: ', image_number)
X_val, y_val = val_generator.__getitem__(image_number)
print(X_val.shape, y_val.shape)

y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
sanity_check_arr(X_val, y_val_argmax,
                note='Val ', 
                batch_size=config['TRAIN']['BATCH_SIZE'])

#%%
print('\nDefine and config model') 
FILTER_NUM=64
dilation_rates = [1, 3, 5, 7, 11, 13]
model = mymodels.Dilated_Attention_ResUNet(input_shape=config['DATA']['DIMENSION'],
                      NUM_CLASSES=2, 
                      FILTER_NUM=FILTER_NUM,
                      # FILTER_SIZE=3,
                      dropout_rate=config['TRAIN']['DROPOUT'], 
                      # batch_norm=True,
                      activation='softmax',
                      dilation_rates=dilation_rates,
                      )
print('Model: ', model.name)
print('FILTER_NUM: ', FILTER_NUM)
print('dilation_rates: ', dilation_rates)

#%%
#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from utility import ( 
                    # dice_coef, dice_coef_loss, jacard_coef,
                    # jacard_coef_loss, iou_coef1, dice_coef1,
                    iou, jaccard_distance, dice_coef2, precision, recall, accuracy)

# dice_loss = sm.losses.DiceLoss()
# focal_loss = sm.losses.CategoricalFocalLoss()
focal_loss = sm.losses.BinaryFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, 
# above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss
# sm_total_loss = sm.losses.categorical_focal_dice_loss  

metrics_list = [
            #'accuracy', 
            sm.metrics.IOUScore(
                        # threshold=0.5, 
                        # class_weights=class_weights
                        ), 
            sm.metrics.FScore(
                        # threshold=0.5, 
                        # class_weights=class_weights
                         ), 
            #iou_coef1, dice_coef1,
            iou, jaccard_distance, dice_coef2, precision, recall, accuracy
            ] 

#%%
model.compile(optimizer=Adam(), 
        # loss=BinaryFocalLoss(gamma=2), # run properly
        # loss=SparseCategoricalFocalLoss(gamma=4), 
        # loss=total_loss, # run properly fold-4
        loss=focal_loss,
        metrics = metrics_list
        )
print('\nloss = (total_loss =  dice_loss + (1 * sm.losses.CategoricalFocalLoss())')
print('\nmetrics = ', metrics_list)

print(model.summary())

#%%
import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)


# print('steps_per_epoch: ', train_steps_RBF)
# print('val_steps_per_epoch: ', val_steps_RBF)

#%
# if LOGSCREEN:
#     verbose=2
# else:
#     verbose=1

# print('\nverbose: ', verbose)

#%%
backup_model_best = os.path.join(
    config['DATA']['SAVEPATH'],
    f'{train_log}.hdf5')

print('\nbackup_model_best: ', backup_model_best)
mcp2 = ModelCheckpoint(backup_model_best,
                       save_best_only=True) 

reLR = ReduceLROnPlateau(
                  # monitor='val_iou_coef1',
                  # monitor='val_dice_coef2',
                  # monitor='val_jaccard_distance', #fold-3 2254
                  monitor='val_loss',
                  factor=0.8,
                  patience=5,
                  verbose=1,
                  mode='auto',
                  #min_lr = 0.00001,#1e-5
                  #min_lr = init_lr/epochs,
                )

early_stop = tf.keras.callbacks.EarlyStopping(
    # Patience should be larger than the one in ReduceLROnPlateau
    patience=10,
    #min_delta=init_lr/epochs
    restore_best_weights = True
    )

print("\nreLR monitor: val_loss 0.8")

#%%
from datetime import datetime 
print("\n\nPerform training...");
print(train_log)
t3 = datetime.now()
with tf.device('/device:GPU:0'):
    model_history = model.fit(
            train_generator, 
            # steps_per_epoch=train_steps_RBF,
            validation_data=val_generator,   
            # validation_steps=val_steps_RBF,
            epochs=config['TRAIN']['EPOCHS'],
            verbose=2,
            callbacks=[
                reLR,
                mcp2,
                early_stop, 
                tqdm_callback, 
                PlotLossesKeras(),
                ],
            )
t4 = datetime.now() - t3
print("\nTraining time: ", t4)

del model

os.mkdir(train_log)
print("Create dir: ", train_log)

#%%
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting

# unet_history_df = pd.DataFrame(unet_history.history) 
# att_unet_history_df = pd.DataFrame(att_unet_history.history) 
model_history_df = pd.DataFrame(model_history.history) 

# with open('unet_history_df.csv', mode='w') as f:
#     unet_history_df.to_csv(f)
    
# with open('att_unet_history_df.csv', mode='w') as f:
#     att_unet_history_df.to_csv(f)

history_file = f'{train_log}_history_df.csv'
history_file_path = os.path.join(train_log, history_file)
with open(history_file_path, mode='w') as f:
    model_history_df.to_csv(f)  
print("\nSaved: ", history_file_path)

#%%
LOGSCREEN=True
if LOGSCREEN==True:
    #Check history plots, one model at a time
    # history1 = unet_history
    # history1 = att_unet_history
    history1 = model_history
    
    #%plot the training and validation accuracy and loss at each epoch
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title('Training and validation loss ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    loss_file = f'loss-{train_log}.png'
    loss_file_path = os.path.join(train_log, loss_file)
    fig.savefig(loss_file_path)
    
    #%%
    acc = history1.history['dice_coef2']
    val_acc = history1.history['val_dice_coef2']
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, acc, 'b', label='Training Dice_coef2')
    plt.plot(epochs, val_acc, 'orange', label='Validation Dice_coef2')
    plt.title('Training and validation Dice_coef ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Dice_coef2')
    plt.legend()
    plt.show()
    dice_file = f'dice_coef2-{train_log}.png'
    dice_file_path = os.path.join(train_log, dice_file)
    fig.savefig(dice_file_path)
    
    #%%
    acc = history1.history['iou']
    val_acc = history1.history['val_iou']
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, acc, 'b', label='Training IoU_coef')
    plt.plot(epochs, val_acc, 'orange', label='Validation IoU_coef')
    plt.title('Training and validation IoU_coef ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('IoU_coef')
    plt.legend()
    plt.show()
    iou_file = f'iou-{train_log}.png'
    iou_file_path = os.path.join(train_log, iou_file)
    fig.savefig(iou_file_path)
    
    #%%
    acc = history1.history['f1-score']
    val_acc = history1.history['val_f1-score']
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, acc, 'b', label='Training F1-score')
    plt.plot(epochs, val_acc, 'orange', label='Validation F1-score')
    plt.title('Training and validation F1-Score ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.legend()
    plt.show()
    f1_file = f'f1_score-{train_log}.png'
    f1_file_path = os.path.join(train_log, f1_file)
    fig.savefig(f1_file_path)
    
    #%%
    lr =  history1.history['lr']
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, lr, 'b', label='Training LR')
    plt.title('Trainining learning rate ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.legend()
    plt.show()
    lr_file = f'LR-{train_log}.png'
    lr_file_path = os.path.join(train_log, lr_file)
    fig.savefig(lr_file_path)
    
    #%%
    acc = history1.history['jaccard_distance']
    val_acc = history1.history['val_jaccard_distance']
    fig = plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(epochs, acc, 'b', label='Training jaccard_distance')
    plt.plot(epochs, val_acc, 'orange', label='Validation jaccard_distance')
    plt.title('Training and validation jaccard_distance ' + train_log)
    plt.xlabel('Epochs')
    plt.ylabel('jaccard_distance')
    plt.legend()
    plt.show()
    f1_file = f'jaccard_distance-{train_log}.png'
    f1_file_path = os.path.join(train_log, f1_file)
    fig.savefig(f1_file_path)

#%%
from tensorflow.keras.models import load_model
print("\nEvaluate model for the val set")
print('\nbackup_model_best: ', backup_model_best)
best_model = load_model(backup_model_best, compile=False)
# print(best_model.summary())
print('\nLoaded: ' , backup_model_best)
best_model.compile(optimizer=Adam(), 
      loss=focal_loss,
      metrics = metrics_list
      )

with tf.device('/device:GPU:0'):
    scores = best_model.evaluate(val_generator, 
                batch_size=config['TRAIN']['BATCH_SIZE'], 
                verbose=1)

#%%
print()     
for metric, value in zip(best_model.metrics_names, scores):
    print("mean {}: {:.4}".format(metric, value))
    
print() 

#%%
image_number = random.randint(0, val_steps)
print('random image number: ', image_number)
X_val, y_val = val_generator.__getitem__(image_number)
print(X_val.shape, y_val.shape)

y_preds = best_model.predict(X_val)
y_preds = np.argmax(y_preds, axis=3)
y_val = np.argmax(y_val, axis=3)

print(y_val.shape, y_preds.shape)

sanity_check_arr(y_val, y_preds, note='Val  ', batch_size=8)

#%%

print(len(best_model.layers))
for index, layer in enumerate(best_model.layers):
    # if layer.name in layer_list:
    print(f"{index}, {layer.name}, {layer.output.shape}\n\n")

# 90, conv2d_155, (None, 32, 32, 1)
# 91, activation_100, (None, 32, 32, 1)
# 92, up_sampling2d_24, (None, 64, 64, 1)

layer_list = [
    'conv2d_155',
    'activation_100',
    'up_sampling2d_24'
    ]


#%%
activation_list = []
for selected_layer_name in layer_list:
    #selected_layer_name = layer_list[i]
    print(selected_layer_name)
    selected_layer = best_model.get_layer(selected_layer_name)
    
    # Create a model that will return the outputs of the selected_layer
    visualization_model = tf.keras.models.Model(
        inputs = best_model.input,
        outputs = [selected_layer.output, best_model.output])
    
    # Preprocess your image and expand its dimensions to match the input shape of the model
    # For example, if your image is loaded in a variable named 'img'
    # preprocessed_img = preprocess_input(img) # preprocess_input is your preprocessing function
    # expanded_img = np.expand_dims(preprocessed_img, axis=0)
    
    # Get the activations
    activations, _ = visualization_model.predict(
        X_val, verbose=1)
    activation_list.append(activations)

#% Visualize the activations
# Each activation map is visualized separately
num_samples = 2 + len(activation_list) # Number of samples, 8 in your case
activations_0 = activation_list[0]
activations_1 = activation_list[1]
activations_2 = activation_list[2]
# activations_3 = activation_list[3]
# activations_4 = activation_list[4]
# activations_pred = np.argmax(activations_4, axis=3)
# Iterate over each sample
for i in range(batch_size):    
    fig = plt.figure(figsize=(18, 8), dpi=300)
    # Plot the original image
    plt.subplot(1, num_samples, 1) # Position in a grid of 3 rows and 'num_samples' columns
    plt.imshow(X_val[i], cmap='gray')
    plt.axis('on') # Hide axis
    plt.title('Image')

    # Plot the corresponding mask
    plt.subplot(1, num_samples, 2 ) # Shift position by 'num_samples'
    (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
    xlabel = str(unique)
    # plt.xlabel(xlabel)
    plt.imshow(y_val_argmax[i], cmap='gray')
    plt.axis('on') 
    plt.title('Mask: ' + xlabel)

    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 3) 
    activation_img = plt.imshow(activations_0[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[0])
    
    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 4) 
    activation_img = plt.imshow(activations_1[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[1])
    
    # Plot the activation for this sample
    ax3 = plt.subplot(1, num_samples, 5) 
    activation_img = plt.imshow(activations_2[i, :, :, 0], 
                cmap='inferno',
                # cmap='gist_heat'
                # cmap='plasma',
                ) # Use index 0 to select the channel
    plt.axis('on') # Hide axis
    plt.title('Attention: ' + layer_list[2])
    
    # Plot the activation for this sample
    # ax3 = plt.subplot(1, num_samples, 6) 
    # activation_img = plt.imshow(activations_3[i, :, :, 0], 
    #             cmap='inferno',
    #             # cmap='gist_heat'
    #             # cmap='plasma',
    #             ) # Use index 0 to select the channel
    # plt.axis('on') # Hide axis
    # plt.title('Attention: ' + layer_list[3])
    
    # # Plot the activation for this sample
    # (unique, counts) = np.unique(activations_pred[i], return_counts=True)
    # pred = str(unique)
    # ax3 = plt.subplot(1, num_samples, 7) 
    # activation_img = plt.imshow(
    #             activations_4[i, :, :, 0],     
    #             cmap='inferno',
    #             # cmap='gist_heat'
    #             # cmap='plasma',
    #             ) # Use index 0 to select the channel
    # plt.axis('on') # Hide axis
    # # plt.title('Attention: ' + layer_list[4])
    # plt.title('Prediction: ' + pred)
   
    # plt.colorbar(activation_img, ax=ax3, orientation='vertical')
    
    fig.tight_layout()   
    plt.show()







