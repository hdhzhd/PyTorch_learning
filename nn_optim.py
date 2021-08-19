# coding ：UTF-8
# 文件功能： 代码实现神经网络优化器的使用
# 开发人员： dpp
# 开发时间： 2021/8/18 4:52 下午
# 文件名称： nn_optim.py
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
optim = torch.optim.SGD(test.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = test(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()   # 将上一轮的每个参数的梯度清零，必须做的，否则梯度计算会出问题
        result_loss.backward()  # 进行反向传播 并计算每个参数的梯度值
        optim.step()    # 对每个参数进行调优
        running_loss = running_loss + result_loss
    print(running_loss)
