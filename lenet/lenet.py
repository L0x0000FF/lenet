import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1,padding=2)# 第一层网络参数
        self.sig = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        # full connected layers
        self.fc1 = nn.Linear(in_features=400,out_features= 120)
        self.fc2 = nn.Linear(in_features=120,out_features= 84)
        self.fc3 = nn.Linear(in_features=84,out_features= 10)
 
    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.pool1(x) 
        x = self.sig(self.conv2(x)) 
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.fc3(x) 
        return x 