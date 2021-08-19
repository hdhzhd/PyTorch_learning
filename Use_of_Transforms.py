# coding ：UTF-8
# 文件功能： 代码实现常用Transforms的功能
# 开发人员： dpp
# 开发时间： 2021/8/11 11:58 上午
# 文件名称： Use_of_Transforms.py
# 开发工具： PyCharm

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/pytorch.jpg").convert('RGB')    #convert('RGB')实现将图片映射到RGB三通道上面
print(img)

# Totensor的使用  ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，并转化成Tensor格式
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor.shape)
writer.add_image("ToTensor", img_tensor)

# Normalize的使用 Normalize(mean，std)是通过下面公式实现数据归一化，mean表示平均值，std表示标准差
# channel=（channel-mean）/std
print(img_tensor[0][0][0])  # 输出第一个像素点归一化之前的值
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 假设均值为[0.5, 0.5, 0.5],标准差为[0.5, 0.5, 0.5]
img_norm = trans_norm(img_tensor)    # 将图片转换为tensor之后，对tensor进行归一化处理
print(img_norm[0][0][0])  # 输出第一个像素点归一化之后的值
writer.add_image("Normalize", img_norm, 1)

# Resize的使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2, 1)

# RandomCrop随机裁剪
trans_random = transforms.RandomCrop(512)  #裁剪图像尺寸为512×512
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):   #对原始图像随机裁剪0-9一共10个图像
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
