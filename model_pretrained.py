# coding ：UTF-8
# 文件功能： 代码实现预训练模型的功能
# 开发人员： dpp
# 开发时间： 2021/8/18 6:45 下午
# 文件名称： model_pretrained.py
# 开发工具： PyCharm
import torchvision

# train_data = torchvision.datasets.ImageNet("ImageNet", split="train", download=True,
#                                           transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# print(vgg16_true)

# 如何利用现有VGG16结构实现CIFAR10的10个类别的输出 在原有VGG16结构后面增加一层线性层
train_data = torchvision.datasets.CIFAR10("CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))   # in_features = 1000 out_features = 10
# print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 修改
print(vgg16_false)
