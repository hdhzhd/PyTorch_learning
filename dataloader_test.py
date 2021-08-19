# coding ：UTF-8
# 文件功能： 代码实现dataloader类的基本功能
# 开发人员： dpp
# 开发时间： 2021/8/13 12:10 上午
# 文件名称： dataloader_test.py.py
# 开发工具： PyCharm
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集  数据放在了CIFAR10文件夹下

test_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 在定义test_loader时，设置了batch_size=4，表示一次性从数据集中取出4个数据
writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1
writer.close()



