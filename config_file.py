import os
import sys
import h5py
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle
import imagesize

config = {}

#paths
#condig['data_folder'] = 'data_224'
config['data_folder'] = 'data_chris/jpeg-melanoma-768x768'#'melanoma-external-malignant-256'
config['save_folder'] = 'example'
config['img_ext'] = '.jpg'
config['train_meta_path'] = 'data_chris/jpeg-melanoma-768x768/train.csv' #'melanoma-external-malignant-256/train_concat.csv'

config['external'] = True
config['external_2018'] = True
config['external_path'] = 'external_data/jpeg-isic2019-768x768/train'
config['external_meta_path'] = 'external_data/jpeg-isic2019-768x768/train.csv'


# meta data
config['sex_ohe'] = True
config['age_normalize'] = True
# training 
config['epochs'] = 20
config['batch_size'] = 6
config['train_meta'] = True
config['freeze_cnn'] = False
config['mixed_prec'] = True
config['test'] = False
#config['upsample'] = 20

#optimizer
config['learning_rate_meta'] = 0.00006
config['learning_rate'] = 0.00006
config['step_decay'] = False
config['plateau_decay'] = True
config['plateau_decay_factor'] = 0.4
config['patience_steps'] = 1
config['CBN'] = False
#loss
config['focalloss'] = False
config['focal_gamma'] = 2
config['focal_alpha'] = 0.75
config['weighted_BCE'] = False
#model 
config['numClasses'] = 1
config['model_type'] = 'efficientnet-b7'
config['fc_layers_before'] = [256,256]
config['dropout_meta'] = 0.2
config['fc_layers_after'] = []
#augmentation
config['TTA'] = 11
config['input_size'] = (768,768)
config['hair_aug'] = False
config['microscope_aug'] = False
config['same_sized_crop'] = False
config['scale_min'] = 0.4
config['full_rot'] = 0
#config['scale'] = (0.8,1.2)
config['cutout'] = True
config['cutout_length'] = 16
config['cutout_hole'] = 1


