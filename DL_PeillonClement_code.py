# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:20:05 2025

@author: peill
"""
import os
# import torch
# import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image 
from pathlib import Path

root_dir = Path(__file__).parent
data_dir = root_dir / 'data'
img_size = 128

def load_imgs(path):
    """
    Load images from local folder

    Parameters
    ----------
    path : path
        Path to the folder containing the images.

    Returns
    -------
    ndarray
        Array containing all the images
        Shape (nb_imgs, img_size, img_size)
    """
    files = os.listdir(path)
    nb_imgs = len(files)
    imgs = np.zeros((nb_imgs, 1, img_size, img_size))
    
    for k, file in enumerate(files):
        img = Image.open(path / file).convert('L') # load image and convert to grayscale
        img = np.asarray(img)
        
        # Add to the list of images
        imgs[k]=img
    return imgs
    
def buildTensor(path):
    damage = load_imgs(path / 'damage')
    no_damage = load_imgs(path / 'no_damage')
    
    m, n = damage.shape[0], no_damage.shape[0]
    
    y0 = np.zeros((m,1))
    y1 = np.ones((n,1))
    
    X = np.concatenate((damage, no_damage), axis=0)
    y = np.concatenate((y0, y1), axis=0)
    
    return X, y
    
# Convert to torch tensors
X_train, y_train = buildTensor(data_dir / 'train_another')
X_val, y_val = buildTensor(data_dir / 'validation_another')
X_test, y_test = buildTensor(data_dir / 'test_another')

















