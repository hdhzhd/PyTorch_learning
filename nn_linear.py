# coding ：UTF-8
# 文件功能： 代码实现神经网络--线性层功能
# 开发人员： dpp
# 开发时间： 2021/8/17 10:28 下午
# 文件名称： nn_linear.py
# 开发工具： PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

test = Test()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    output = torch.flatten(imgs)
    print(output.shape)  # torch.Size([1, 1, 1, 196608])
    output = test(output)
    print(output.shape)  # torch.Size([1, 1, 1, 10])
