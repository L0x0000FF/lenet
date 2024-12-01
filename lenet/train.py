from torchvision.datasets import MNIST
import numpy as np
from torchvision import transforms 
import torch.utils.data as Data
from lenet import LeNet
from datetime import time
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

def getDataLoader():
    train_data = MNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
 
    # split the dataset
    train_data, val_data = Data.random_split(train_data,lengths=[round(0.8*len(train_data)),round(0.2*len(train_data))])
    
    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True,
                                        num_workers=2)
    val_data_loader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True,
                                       num_workers=2)
 
    return train_data_loader, val_data_loader
 
def train(model, train_data_loader, val_data_loader,num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = {name: param.clone().detach() for name, param in model.named_parameters()}

    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
 
    since = time.time()
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0
        for step,(b_x, b_y) in enumerate(train_data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
 
            model.train()
            
            output = model(b_x)
 
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
 
            optimizer.zero_grad()
            
            loss.backward()
 
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y)
            train_num += b_x.size(0)
 
        for step, (b_x, b_y) in enumerate(val_data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
 
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_acc += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append((train_acc / train_num).cpu())
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append((val_acc / val_num).cpu())
        
        print("{}Train Loss:{:.4f} Train Acc:{:.4f} Test Loss:{:.4f} Test Acc:{:.4f}".format(
            epoch, train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]
        ))
 
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = {name: param.clone().detach() for name, param in model.named_parameters()}
        time_use = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_use//60, time_use%60))
 
 
    torch.save(best_model_wts, './param.pth')
    train_process = [
        range(num_epochs),
        train_loss_all,
        train_acc_all,
        val_loss_all,
        val_acc_all
    ]
    return train_process
 
def matplot_acc_loss(train_process):
    plt.figure(1)
    plt.title('loss')
    plt.plot(train_process[0], train_process[1], 'ro-',label='train loss')
    plt.plot(train_process[0], train_process[3], 'bs-',label='validate loss')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
 
    plt.figure(2)
    plt.title('accuracy')
    plt.plot(train_process[0], train_process[2], 'ro-',label='train accuracy')
    plt.plot(train_process[0], train_process[4], 'bs-',label='validate accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
 
def export2onnx(model,batch_size,channel,input_features,output_features):
    model.to('cpu')
    input_data = torch.rand((batch_size,channel,input_features[0],input_features[1]))
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(model,input_data,"LeNet_Static.onnx",verbose=True,input_names=[input_name],output_names=[output_name])

if __name__ == '__main__':
    model = LeNet()
    train_data_loader, val_data_loader = getDataLoader()
    train_process = train(model, train_data_loader, val_data_loader, 20)
    matplot_acc_loss(train_process)