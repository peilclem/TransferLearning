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
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
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
EPOCHS = 3

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
            label = self.target_transform(label)                               # scale values between 0 and 1            
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
print(f'\nSize of a single image: {train_features[0].shape}\n')                  # should be (3, 128, 128)

#%% Build model --> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

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
        self.flatten = nn.Flatten(1,-1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # print(f'Layer1 \t\tout_shape: {x.shape}')
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # print(f'Layer2 \t\tout_shape: {x.shape}')
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # print(f'Layer3 \t\tout_shape: {x.shape}')
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        # print(f'Layer4 \t\tout_shape: {x.shape}')
        x = self.flatten(x)
        # print(f'Flatten\t\tout_shape: {x.shape}')
        
        x = self.fc1(x)
        x = F.relu(x)
        # print(f'Linear1 \tout_shape: {x.shape}')
        x = self.fc2(x)
        x = F.sigmoid(x)
        # print(f'Output \t\tout_shape: {x.shape}')
        return x


model = MyCNN()
print(model)

criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001,)


#%% TRAINING LOOOP
def calculate_accuracy(y_gt, y_pred):
    predictions = torch.round(y_pred)
    correct = (predictions == y_gt).float().sum()
    total = y_gt.shape[0]
    accuracy = correct / total
    return accuracy

print('-'*30)
print(f'{"Start Training":^30}')
print('-'*30)
    
for epoch in range(EPOCHS): # 3'/epoch on CPU --> try to train on cluster
    loss_tab, loss_batch = [], 0
    accuracy_tab, accuracy_batch = [], 0
    count = 0
    for img, y_gt in tqdm(train_dataloader):
        count += 1
        y_gt = y_gt.reshape(-1,1).float()
        
        optimizer.zero_grad()
        
        y_pred = model(img)
        loss = criterion(y_pred, y_gt)     
        loss.backward()
        optimizer.step()
        
        loss_batch += loss.item()
        accuracy_batch += calculate_accuracy(y_gt, y_pred)
    
    # Store loss value at epoch end
    loss_tab.append(loss_batch)
    accuracy_tab.append(accuracy_batch/count)
    
    print(f'[{epoch:03d}/{EPOCHS}]\tLoss: {loss_batch:.6f}\tAccuracy: {accuracy_batch/count*100:.2f}%\n')        
        
print('-'*30)
print(f'{"End Training":^30}')
print('-'*30)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        