from torchvision.datasets import MNIST # 获取数据集
import numpy as np
from torchvision import transforms # 归一化数据集
import torch.utils.data as Data
from lenet import LeNet
from datetime import time
import torch
import time
import pandas as pd
import cv2

train_data = MNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                            download=True)
dataLoader = Data.DataLoader(train_data,batch_size=1)
ditr = iter(dataLoader)
imgs = []
model = torch.load('lenet.pt')
for i in range(10):
  img, label = next(ditr)
  img_ = torch.squeeze(img.detach()).numpy()
  imgs.append(img_)
imgs = np.hstack(imgs)
cv2.imshow("1",imgs)
if cv2.waitKey(0) == 'q':
  cv2.destroyAllWindows()