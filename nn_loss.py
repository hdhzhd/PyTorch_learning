# coding ：UTF-8
# 文件功能： 代码实现损失函数loss function功能
# 开发人员： dpp
# 开发时间： 2021/8/18 12:12 下午
# 文件名称： nn_loss.py
# 开发工具： PyCharm
import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

loss_mse = MSELoss(reduction="sum")
result_mse = loss_mse(inputs,targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # 是一个3分类问题，所以reshape里面有3
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
