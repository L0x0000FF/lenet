from lenet import LeNet
from torchvision.datasets import FashionMNIST
import torch
import torch.utils.data as Data
from torchvision import transforms
 
# 处理测试集
def test_data_process():
    # 定义数据集
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
 
    test_data_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True,
                                        num_workers=2)
 
    return test_data_loader
 
def test_model_process(model, test_data_loader):
    # 设备：cup或者gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    # 参数初始化
    test_correct = 0
    test_num = 0
 
    # 只进行前向传播，不进行梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_data_loader:
            test_data_x = test_data_x.to(device) # 将数据传输设备
            test_data_y = test_data_y.to(device) # 将标签传入设备
 
            # 测试模式
            model.eval()
            # 前向传播
            output = model(test_data_x)
            # 查找每一行最大值下标
            pre_lab = torch.argmax(output, dim=1)
            # 精确度总和
            test_correct += torch.sum(pre_lab == test_data_y)
            # 样本数总和
            test_num += test_data_x.size(0)
 
    # 计算测试准确率
    test_acc = test_correct.double().item() / test_num
 
    print("测试准确率：", test_acc)
 
 
# 模型测试
if __name__ == '__main__':
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('param.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
    # 获取数据集
    test_data_loader = test_data_process()
    # 模型测试
    #test_model_process(model, test_data_loader)
 
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_data_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
 
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
 
            print("预测值：", classes[result],"-----------  真实值", classes[label])