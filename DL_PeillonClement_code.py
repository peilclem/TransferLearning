# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:20:05 2025

@author: peill
"""
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
        image = Image.open(img_path)                                           # use pillow to avoid having a tensor 
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)                                                 # scale values between 0 and 1            
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

# To transform the images in grayscale
transform = transforms.ToTensor()

train_dataset = CustomImageDataset(data_dir / 'train_another', transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CustomImageDataset(data_dir / 'validation_another', transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomImageDataset(data_dir / 'test_another', transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features[0])
print(train_features[0].shape)                                                 # should be (3, 128, 128)

#%% Build model --> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3,3), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        print(f'Layer1 \t\tout_shape: {x.shape}')
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        print(f'Layer2 \t\tout_shape: {x.shape}')
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        print(f'Layer3 \t\tout_shape: {x.shape}')
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        print(f'Layer4 \t\tout_shape: {x.shape}')
        x = torch.flatten(x, 0)
        print(f'Flatten\t\tout_shape: {x.shape}')
        
        x = self.fc1(x)
        x = F.relu(x)
        print(f'Linear1 \tout_shape: {x.shape}')
        x = self.fc2(x)
        x = F.sigmoid(x)
        print(f'Output \t\tout_shape: {x.shape}')
        return x

#%%
model = MyCNN()   
x = train_features[0]
x.shape
y = model.forward(x)
print(y.item())
