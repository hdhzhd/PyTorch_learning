# coding ：UTF-8
# 文件功能： 代码实现神经网络模型的保存功能
# 开发人员： dpp
# 开发时间： 2021/8/18 11:30 下午
# 文件名称： model_save.py
# 开发工具： PyCharm
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1:不仅保存了网络模型，也保存了网络模型中的相关参数
torch.save(vgg16, "vgg16_model1.pth")

# 保存方式2：只保存了模型的参数，占用空间更小，官方推荐方式
torch.save(vgg16.state_dict(), "vgg16_model2.pth")
