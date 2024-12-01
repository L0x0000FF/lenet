from torchvision.datasets import MNIST # 获取数据集
import numpy as np
from torchvision import transforms # 归一化数据集
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
    # 复制当前模型参数
    best_model_wts = {name: param.clone().detach() for name, param in model.named_parameters()}
    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    #  训练集损失列表
    train_loss_all = []
    # 测试集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 测试集准确度列表
    val_acc_all = []
 
    # 当前时间
    since = time.time()
 
    # 按轮次训练模型
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集精度
        train_acc = 0.0
        # 测试集损失
        val_loss = 0.0
        # 测试集精度
        val_acc = 0.0
        # 样本数量
        train_num = 0
        val_num = 0
        for step,(b_x, b_y) in enumerate(train_data_loader):
            b_x = b_x.to(device)# 将特征放到设备里面
            b_y = b_y.to(device)# 将标签放入设备里面
 
            model.train()# 模型打开训练模式
            # 将数据放入模型进行前向传播
            output = model(b_x)
 
            # softmax 分类器，得到最大概率值 --- 即分类标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失----根据标签 --- 是一个张量，表示差异程度
            loss = criterion(output, b_y)
 
            # 将梯度初始化为0 防止梯度积累
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
 
            # 更新网络参数
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，准确度加1
            train_acc += torch.sum(pre_lab == b_y)
            # 获取样本数
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
        # 计算并保存每一次迭代的loss值和精确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append((train_acc / train_num).cpu())
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append((val_acc / val_num).cpu())
        #打印
        print("{}Train Loss:{:.4f} Train Acc:{:.4f} Test Loss:{:.4f} Test Acc:{:.4f}".format(
            epoch, train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]
        ))
 
        if val_acc_all[-1] > best_acc:
            # 保存当前最优准确度
            best_acc = val_acc_all[-1]
            # 保存当前最优参数
            best_model_wts = {name: param.clone().detach() for name, param in model.named_parameters()}
        # 计算运行时间
        time_use = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_use//60, time_use%60))
 
 
    # 选择最优参数 加载最高准确率下的模型参数
    torch.save(best_model_wts, './param.pth')
    # 将数据转换为dataframe 用于最后的可视化
    train_process = [
        range(num_epochs),
        train_loss_all,
        train_acc_all,
        val_loss_all,
        val_acc_all
    ]
    return train_process
 
# 可视化loss和精确度
def matplot_acc_loss(train_process):
    # print(train_process)
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
    # 将模型实例化
    model = LeNet()
    # 得到数据
    train_data_loader, val_data_loader = getDataLoader()
    # 训练模型
    train_process = train(model, train_data_loader, val_data_loader, 20)
    # 画图
    matplot_acc_loss(train_process)
    torch.save(model.state_dict(),"lenet.pth")
    # export2onnx(model=model,batch_size=64,channel=1,input_features=(28,28),output_features=10)
    