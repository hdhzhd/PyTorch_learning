# coding ：UTF-8
# 文件功能： s
# 开发人员： dpp
# 开发时间： 2021/8/16 4:18 下午
# 文件名称： nn_conv2d.py
# 开发工具： PyCharm

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        # 定义卷积操作相关参数 输入通道3 输出通道6 卷积核大小3 步长1 原图不进行边缘填充
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

test = Test()
print(test)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    print(imgs.shape)    # 输出原始图像的形状  torch.Size([64, 3, 32, 32])
    print(output.shape)  # 输出卷积之后图像的形状  torch.Size([64, 6, 30, 30])
    # 原始图像的形状：torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # 卷积之后的图像的形状：torch.Size([64, 6, 30, 30])  --> [xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
