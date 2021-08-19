# coding ：UTF-8
# 文件功能： 代码实现PyTorch中dataset和transforms综合使用的功能
# 开发人员： dpp
# 开发时间： 2021/8/11 5:42 下午
# 文件名称： dataset_transforms.py
# 开发工具： PyCharm
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# train=True表明是训练数据集 transform=dataset_transform表示将PIL的图像转换成tensor格式的 download=True表示下载该数据集
train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=dataset_transform, download=True)
# train=False表明是测试数据集 transform=dataset_transform表示将PIL的图像转换成tensor格式的，download=True表示下载该数据集
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()

