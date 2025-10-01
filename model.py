import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

data_dir = '/home/muri/Documents/PI'

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Primeira camada convolucional
        self.pool = nn.MaxPool2d(2, 2)                          # Max Pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32 * 16 * 16, 128)                 # Flatten -> Fully Connected
        self.dropout = nn.Dropout(0.5)                          # Dropout
        self.fc2 = nn.Linear(128, num_classes)                  # Saída final

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
