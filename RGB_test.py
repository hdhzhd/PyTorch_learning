# coding ：UTF-8
# 文件功能： 代码实RGB颜色空间基本功能
# 开发人员： dpp
# 开发时间： 2021/8/11 8:50 下午
# 文件名称： RGB_test.py
# 开发工具： PyCharm
from skimage import data
from matplotlib import pyplot as plt
image = data.coffee()
fig = plt.figure()

# 显示RGB图像
plt.figure()
plt.axis('off')
plt.imshow(image)
plt.show()

# 显示R通道图像
imageR = image[:, :, 0]
plt.figure()
plt.axis('off')
plt.imshow(imageR, cmap='gray')
plt.show()

# 显示G通道图像
imageG = image[:, :, 1]
plt.figure()
plt.axis('off')
plt.imshow(imageG, cmap='gray')
plt.show()

# 显示B通道图像
imageB = image[:, :, 2]
plt.figure()
plt.axis('off')
plt.imshow(imageB, cmap='gray')
plt.show()
