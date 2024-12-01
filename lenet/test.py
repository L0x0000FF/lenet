from lenet import LeNet
from torchvision.datasets import MNIST
import torch
import torch.utils.data as Data
from torchvision import transforms
 
def test_data_process():
    test_data = MNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
 
    test_data_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True,
                                        num_workers=2)
 
    return test_data_loader
 
def test_model_process(model, test_data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
 
    test_correct = 0
    test_num = 0
 
    with torch.no_grad():
        for test_data_x, test_data_y in test_data_loader:
            test_data_x = test_data_x.to(device) 
            test_data_y = test_data_y.to(device) 
 

            model.eval()

            output = model(test_data_x)
            
            pre_lab = torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y)

            test_num += test_data_x.size(0)
            print("predicted:",pre_lab.cpu().detach().numpy()," test data:",test_data_y.cpu().detach().numpy())
 
    test_acc = test_correct.double().item() / test_num
 
    print("Accuracy:", test_acc)
 
 
if __name__ == '__main__':

    model = LeNet()
    model.load_state_dict(torch.load('param.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))

    test_data_loader = test_data_process()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model_process(model=model,test_data_loader=test_data_loader)