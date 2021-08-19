# coding ：UTF-8
# 文件功能： 代码实现搭建神经网络，并结合Sequential的使用
# 开发人员： dpp
# 开发时间： 2021/8/17 11:31 下午
# 文件名称： nn_sequential.py
# 开发工具： PyCharm
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


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
print(test)  # 输出网络的结构情况

input = torch.ones((64, 3, 32, 32))
output = test(input)
print(output.shape)  # 输出output的尺寸

writer = SummaryWriter("logs")
writer.add_graph(test, input)
writer.close()
