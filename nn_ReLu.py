# coding ：UTF-8
# 文件功能： 代码实现非线性激活功能
# 开发人员： dpp
# 开发时间： 2021/8/17 8:41 下午
# 文件名称： nn_ReLu.py
# 开发工具： PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

test = Test()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = test(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
