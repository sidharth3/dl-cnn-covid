import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

def weight_initialization(layer):
    if layer == nn.Conv2d or layer == nn.Linear:
        torch.nn.init.kaiming_normal(layer.weight, nonlinearity = 'relu')

class Binary_Classifier_One(nn.Module):
    
    def __init__(self):
        super(Binary_Classifier_One,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        
        self.adap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
#         self.fc3 = nn.Linear(32, 1)
        self.conv1.apply(weight_initialization)
        self.conv2.apply(weight_initialization)
        self.conv3.apply(weight_initialization)
        self.conv4.apply(weight_initialization)
        self.fc1.apply(weight_initialization)
        self.fc2.apply(weight_initialization)
#         self.fc3.apply(weight_initialization)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adap(x)
        x = self.fc1(x)
        x = self.fc2(x)
#         x = self.fc3(x)
        return torch.sigmoid(x)
  
class Binary_Classifier_Two(nn.Module):
    
    def __init__(self):
        super(Binary_Classifier_Two,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        
        self.adap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.conv1.apply(weight_initialization)
        self.conv2.apply(weight_initialization)
        self.conv3.apply(weight_initialization)
        self.conv4.apply(weight_initialization)
        self.conv5.apply(weight_initialization)
        self.conv6.apply(weight_initialization)
        self.fc1.apply(weight_initialization)
        self.fc2.apply(weight_initialization)
        self.fc3.apply(weight_initialization)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.adap(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)
    

#TODO: Three-Way_Classifier
class Three_Way_Classifier_One(nn.Module):
    def __init__(self):
        super(Three_Way_Classifier_One,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.05))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.05))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.05))

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 3)
        self.conv1.apply(weight_initialization)
        self.conv2.apply(weight_initialization)
        self.conv3.apply(weight_initialization)
        self.conv4.apply(weight_initialization)
        self.fc1.apply(weight_initialization)
        self.fc2.apply(weight_initialization)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
