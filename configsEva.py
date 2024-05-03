#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:08 2024

@author: kasikritdamkliang
"""
import os, platform
from datetime import datetime 
from yacs.config import CfgNode as CN

# Define default configuration
_C = CN()

# Base config files
# _C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

if platform.system() == 'Windows':
    _C.BASE = str(platform.system())
    _C.BASEPATH = 'E:\\Tongue'
    
    _C.DATA.TRAIN_PATH = os.path.join(_C.BASEPATH,
                        'dataset',
                        'Crop_images',
                        '00')
    _C.DATA.SET = 'A' #crop
    _C.DATA.DEVICE = 'DSLR'
    
    # _C.DATA.TRAIN_PATH = os.path.join(_C.BASEPATH,
    #         'dataset',
    #         'Crop_images',
    #         '01')
    # _C.DATA.SET = 'A' # Crop
    # _C.DATA.DEVICE = 'Mobile'
      
    # _C.DATA.TRAIN_PATH = os.path.join(_C.BASEPATH,
    #                     'dataset',
    #                     'ROI_images', #B
    #                     '00',
    #                     'image')
    # _C.DATA.SET = 'B'
    # _C.DATA.DEVICE = 'DSLR'
    
    # _C.DATA.TRAIN_PATH = os.path.join(_C.BASEPATH,
    #         'dataset',
    #         'ROI_images',
    #         '01',
    #         'image')
    # _C.DATA.SET = 'B'
    # _C.DATA.DEVICE = 'Mobile'
    
    # _C.DATA.SET = 'C'
    # _C.DATA.DEVICE = 'DSLR'
    # _C.DATA.IMAGE = 'E:\\Tongue\\dataset\\Crop_images\\00'
    # _C.DATA.MASK = 'E:\\Tongue\\dataset\\ROI_images\\00\\mask'
      
    # _C.DATA.SAVEPATH = os.path.join(_C.BASEPATH, 'models')
    _C.DATA.SAVEPATH = os.path.join(_C.BASEPATH, 'model-best')
    
elif platform.system() == 'Darwin':
    _C.BASE = str(platform.system())
    _C.BASEPATH = '/Users/kasikritdamkliang/Datasets/Tongue'
    
    _C.DATA.SET = 'A'
    _C.DATA.TRAIN_PATH = os.path.join(_C.BASEPATH,
                        'dataset',
                        'Crop_images',
                        '00')
    _C.DATA.DEVICE = 'DSLR' 
    
    # _C.DATA.SET = 'C'
    # _C.DATA.DEVICE = 'DSLR'
    # _C.DATA.IMAGE = os.path.join(_C.BASEPATH,
    #                     'dataset/Crop_images/00')
    # _C.DATA.MASK = os.path.join(_C.BASEPATH,
    #                     'dataset/ROI_images/00/mask')
       
    _C.DATA.SAVEPATH = os.path.join(_C.BASEPATH,
                        'models')
    
else: #Linux
    _C.DATA.SAVEPATH = 'models'
    _C.DATA.SET = 'C'
        
    if _C.DATA.SET == 'C':
        _C.DATA.IMAGE = 'dataset/Crop_images/00/image'
        _C.DATA.MASK = 'dataset/ROI_images/00/mask'
    else:
        _C.DATA.TRAIN_PATH = 'ROI_images/00/image'
        _C.DATA.SET = 'B'
        
    
# _C.DATA.FEATURE_PATH = '2024-03-15-PJ.csv'
_C.DATA.FEATURE_PATH = '20240324-3TTM.csv'
_C.DATA.KAPPA_PATH = _C.DATA.FEATURE_PATH
# _C.DATA.KAPPA_PATH = '2024-03-15.csv'

# _C.DATA.SEED = 1337
# _C.DATA.SEED = 42
# _C.DATA.SEED = 2024

_C.DATA.SEEDS = [
    # 1337,
    # 42,
    2024
    ]

_C.DATA.C = 3
# _C.DATA.W = _C.DATA.H = 256
_C.DATA.W = _C.DATA.H = 299
# _C.DATA.W = _C.DATA.H = 384
# _C.DATA.W = _C.DATA.H = 1024
_C.DATA.IMAGE_SIZE = _C.DATA.W
_C.DATA.DIMENSION = _C.DATA.W, _C.DATA.H, _C.DATA.C
_C.DATA.TEST_SIZE = 0.3

_C.MODEL = CN()

_C.EVALUATE = True

if _C.DATA.W == 384:
    _C.MODEL.NAMES = ["ViT_b16",]
    
elif _C.DATA.W == 299:    
    _C.MODEL.NAMES = [
        # 'VGG16',
        # 'VGG19',                      
        'DenseNet121',
        # 'MobileNet',
        # 'MobileNetV2',       
        # 'InceptionResNetV2',  
        # "EfficientNetB7", 
        # 'Xception',  #exhousted
        # 'InceptionV3', #exhousted
        ]   
    
else:
    _C.MODEL.NAMES = ["TL-DARUN"]

_C.MODEL.NUM_CLASSES = 3

_C.TRAIN = CN()
# _C.TRAIN.TL = True
_C.TRAIN.TL = False
# _C.TRAIN.SAVEMODEL = True



_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")
_C.TRAIN.BATCH_SIZE = 8
# _C.TRAIN.LR = 1e-5
_C.TRAIN.EPOCHS = 120
_C.TRAIN.CLASS_LABELS = ['Pitta', 'Vata', 'Kapha']
_C.TRAIN.CLASSES = [0, 1, 2]
_C.TRAIN.CLASSDICT = [(0, "Pitta"), (1, "Vata"), (2, "Kapha")]
_C.TRAIN.DROPOUT = 0.25


if _C.TRAIN.TL == True and _C.DATA.W == 299:
    # _C.MODEL.BEST = 'Windows-A-DSLR-ViT-20240325-1832.hdf5'
    # _C.MODEL.BEST = 'Windows-A-DSLR-VGG16-20240325-1716.hdf5'
    # _C.MODEL.BEST = 'Windows-A-DSLR-DenseNet121-20240403-1546.hdf5'
    # _C.MODEL.BEST = 'Windows-B-seed1337-DSLR-MobileNet-20240404-1635.hdf5'
    # _C.MODEL.BEST = 'Windows-A-DSLR-InceptionResNetV2-20240403-1802.hdf5'
    # _C.MODEL.BEST = 'Windows-A-DSLR-VGG19-20240403-1238.hdf5'
    # _C.MODEL.BEST = 'Windows-A-DSLR-Xception-20240403-1259.hdf5'
    _C.MODEL.BEST = 'Windows-A-seed42-DSLR-Xception-20240419-1022.hdf5'
    
    # _C.MODEL.BEST = 'Windows-B-seed1337-DSLR-InceptionResNetV2-20240404-1635.hdf5'
    # _C.MODEL.BEST = 'Windows-B-seed2024-DSLR-InceptionV3-20240419-0929.hdf5'
    # _C.MODEL.BEST = 'Windows-B-seed42-DSLR-Xception-20240418-1443.hdf5'
    # _C.MODEL.BEST = 'Windows-B-seed42-DSLR-MobileNet-20240418-1605.hdf5'
    # _C.MODEL.BEST = 'Windows-B-seed1337-DSLR-MobileNetV2-20240404-1635.hdf5'
    
if _C.TRAIN.TL == True and _C.DATA.W == 384:
    _C.MODEL.BEST =  'Windows-B-seed2024-DSLR-ViT-20240418-1402.hdf5'
    

if _C.DATA.SET == 'C':
    _C.MODEL.NAMES = ['DARUN']
    
_C.SAVE = False
_C.SAVEHIST = False

    
def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    #update_config(config, args)

    return config

