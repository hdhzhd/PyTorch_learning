# coding ：UTF-8
# 文件功能： 代码实现模型定义功能
# 开发人员： dpp
# 开发时间： 2021/8/19 11:31 上午
# 文件名称： model.py
# 开发工具： PyCharm

import torch
from torch import nn

# 搭载神经网络
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ =='__main__':
    test = Test()
    input = torch.ones((64, 3, 32, 32))
    output = test(input)
    print(output.shape)