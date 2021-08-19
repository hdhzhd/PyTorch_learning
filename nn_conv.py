# coding ：UTF-8
# 文件功能： 代码实现卷积操作的基本功能
# 开发人员： dpp
# 开发时间： 2021/8/16 3:56 下午
# 文件名称： nn_conv.py
# 开发工具： PyCharm

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                      [0, 1, 0],
                      [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)

output_2 = F.conv2d(input, kernel, stride=2)
print(output_2)

output_3 = F.conv2d(input, kernel, stride=1, padding=1)   # 在输入图像周围扩展一个像素
print(output_3)
