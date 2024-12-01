from torchvision.datasets import MNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
from lenet import LeNet
from datetime import time
import torch
import time
import pandas as pd
from matplotlib import pyplot as plt

train_data = MNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                            download=True)
dataLoader = Data.DataLoader(train_data,batch_size=1)
ditr = iter(dataLoader)
imgs = []
model = torch.load('param.pth')
for i in range(10):
  img, label = next(ditr)
  img_ = torch.squeeze(img.detach()).numpy()
  imgs.append(img_)
imgs = np.hstack(imgs)
plt.imshow(imgs)
plt.show()