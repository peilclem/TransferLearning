# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:20:05 2025

@author: peill
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image 
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

root_dir = Path(__file__).parent
data_dir = root_dir / 'data'

# Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 64

# def load_imgs(path):
#     """
#     Load images from local folder

#     Parameters
#     ----------
#     path : path
#         Path to the folder containing the images.

#     Returns
#     -------
#     ndarray
#         Array containing all the images
#         Shape (nb_imgs, IMG_SIZE, IMG_SIZE)
#     """
    
#     files = os.listdir(path)
#     nb_imgs = len(files)
#     imgs = np.zeros((nb_imgs, 1, IMG_SIZE, IMG_SIZE))
    
#     for k, file in enumerate(files):
#         img = Image.open(path / file).convert('L') # load image and convert to grayscale
#         img = np.asarray(img)
        
#         # Add to the list of images
#         imgs[k]=img
#     return imgs
    
# def buildTensor(path):
#     damage = load_imgs(path / 'damage')
#     no_damage = load_imgs(path / 'no_damage')
    
#     m, n = damage.shape[0], no_damage.shape[0]
    
#     y0 = np.zeros((n,1))
#     y1 = np.ones((m,1))
    
#     X = np.concatenate((damage, no_damage), axis=0)
#     y = np.concatenate((y0, y1), axis=0)
    
#     return torch.Tensor(X), torch.Tensor(y)
    
# # Convert to torch tensors
# X_train, y_train = buildTensor(data_dir / 'train_another')
# X_val, y_val = buildTensor(data_dir / 'validation_another')
# X_test, y_test = buildTensor(data_dir / 'test_another')


# Create dataset & Dataloader --> https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):    
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = self.build_annotationFile()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def build_annotationFile(self):
        damage = os.listdir(self.img_dir / 'damage')
        no_damage = os.listdir(self.img_dir / 'no_damage')
        
        m = len(damage)
        n = len(no_damage)
        
        annotation_file = []
        
        for k, file in enumerate(damage+no_damage):
            if k < m:
                annotation_file.append([str(self.img_dir / 'damage' / file), 1])
            else:
                annotation_file.append([str(self.img_dir / 'no_damage' / file), 0])

        return pd.DataFrame(annotation_file)

train_dataset = CustomImageDataset(data_dir / 'train_another')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CustomImageDataset(data_dir / 'validation_another')
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomImageDataset(data_dir / 'test_another')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

""" TRANSFORM INTO GRAYSCALE IMG"""
train_features, train_labels = next(iter(train_dataloader))
train_features[0].shape
