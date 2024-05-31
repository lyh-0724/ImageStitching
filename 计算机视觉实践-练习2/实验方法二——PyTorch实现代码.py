import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义卷积层C1，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        # 定义池化层S2，池化核大小为2x2，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义卷积层C3，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 定义池化层S4，池化核大小为2x2，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义全连接层F5，输入节点数为16x4x4=256，输出节点数为120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 定义全连接层F6，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义输出层，输入节点数为84，输出节点数为10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层C1
        x = self.conv1(x)
        # print('卷积层C1后的形状:', x.shape)
        # 池化层S2
        x = self.pool1(torch.relu(x))
        # print('池化层S2后的形状:', x.shape)
        # 卷积层C3
        x = self.conv2(x)
        # print('卷积层C3后的形状:', x.shape)
        # 池化层S4
        x = self.pool2(torch.relu(x))
        # print('池化层S4后的形状:', x.shape)
        # 全连接层F5
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        # print('全连接层F5后的形状:', x.shape)
        x = torch.relu(x)
        # 全连接层F6
        x = self.fc2(x)
        # print('全连接层F6后的形状:', x.shape)
        x = torch.relu(x)
        # 输出层
        x = self.fc3(x)
        # print('输出层后的形状:', x.shape)
        return x


# 设置超参数
batch_size = 64
learning_rate = 0.01
epochs = 100

# 准备数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型和优化器
model = LeNet5()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 定义模型保存路径和文件名
model_path = 'model.pth'
if os.path.exists(model_path):
    # 存在，直接加载模型
    model.load_state_dict(torch.load(model_path))
    print('Loaded model from', model_path)
else:
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            # 将图像展平
            # images = images.view(images.size(0), -1)
            images = images.view(-1, 1, 28, 28)
            # 将数据放入模型
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        # 在测试集上测试模型
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                # 将图像展平
                images = images.view(-1, 1, 28, 28)
                # 将数据放入模型
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / len(test_dataset)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, loss.item(), accuracy))

    torch.save(model.state_dict(), 'model.pth')

for i in range(10):
    img, label = next(iter(test_loader))
    img = img[i].unsqueeze(0)

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(img)

    # 解码预测结果
    pred = output.argmax(dim=1).item()
    print(f'Predicted class: {pred}, actual value: {label[i]}')


