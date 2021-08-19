# coding ：UTF-8
# 文件功能： 代码实现神经网络--池化层功能
# 开发人员： dpp
# 开发时间： 2021/8/17 4:53 下午
# 文件名称： nn_maxpool.py
# 开发工具： PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

test = Test()

step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = test(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
