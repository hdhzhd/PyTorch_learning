# coding ：UTF-8
# 文件功能： 代码实现PyTorch中Transforms基本功能
# 开发人员： dpp
# 开发时间： 2021/8/9 11:32 下午
# 文件名称： main.py
# 开发工具： PyCharm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法-》tensor数据类型
# 通过transforms.ToTensor去解决两个问题

# 2、为什么我们需要Tensor数据类型

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# writer用于tensorboard显示图片
writer = SummaryWriter('logs')

# 1、transsforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()