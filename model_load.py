# coding ：UTF-8
# 文件功能： 代码实现神经网络模型的加载功能
# 开发人员： dpp
# 开发时间： 2021/8/18 11:33 下午
# 文件名称： model_load.py
# 开发工具： PyCharm
import torch

# 方式1-> 保存方式1，加载模型
import torchvision

model = torch.load("vgg16_model1.pth")
print(model)

# 方式2-> 保存方式2，加载模型
model = torch.load("vgg16_model2.pth")  # 加载出来的是字典类型的数据
print(model)

# 方式2-> 保存方式2，加载模型结构
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_model2.pth"))  # 输出完整的模型结构，与第一种方式输出的模型结构相同
print(vgg16)