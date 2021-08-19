# coding ：UTF-8
# 文件功能： 代码实现nn.module模块基本使用的功能
# 开发人员： dpp
# 开发时间： 2021/8/15 11:18 下午
# 文件名称： nn_module.py
# 开发工具： PyCharm
import torch
from torch import nn

class Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

test = Test()
x = torch.tensor(1.0)
output = test(x)
print(output)
