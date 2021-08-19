# coding ：UTF-8
# 文件功能： 代码实现在神经网络中使用损失函数的功能
# 开发人员： dpp
# 开发时间： 2021/8/18 3:52 下午
# 文件名称： nn_loss_network.py
# 开发工具： PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

test = Test()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = test(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
