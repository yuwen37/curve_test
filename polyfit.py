import numpy as np
from PIL import Image
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

img_path = 'semicircle.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, BW = cv2.threshold(gray, 80, 1, cv2.THRESH_BINARY)
plt.figure(1)
plt.imshow(BW, cmap='binary')
plt.show()
plt.close(1)

# 提取边缘
y, x = np.where(BW == 0)
df = pd.DataFrame({'x': x, 'y': y})
data_group = df.groupby(['x']).max()
data_group.index.name = 'index'
data_group['x'] = data_group.index
data_group.sort_values('x', ascending=True, inplace=True)
x = np.array(data_group['x']).astype(np.int)
y = np.array(data_group['y']).astype(np.int)
# plt.figure(2)
# plt.plot(x, y)
# plt.show()
# plt.close(2)

# 提取比例因子
factor = 2/(max(x) - min(x))

# 计算曲率
x = x * factor
y = y * factor
plt.figure(2)
plt.plot(x, y)
plt.show()
plt.close(2)

# 多项式拟合
length = len(x)
delta = 30
K = np.zeros(length)
for i in range(delta, length-delta):
    # 2次多项式
    # c0 = np.polyfit(x[i - delta:i + delta], y[i - delta:i + delta], 2)
    # c1 = np.array([c0[0]*2, c0[1]*1])
    # c2 = np.array([c1[1]])

    # 3次多项式
    c0 = np.polyfit(x[i - delta:i + delta], y[i - delta:i + delta], 3)  # 挠曲线系数
    c1 = np.array([c0[0] * 3, c0[1] * 2, c0[2]])  # 1阶导函数系数
    c2 = np.array([c1[0] * 2, c1[1]])  # 2阶导函数系数

    # 4次多项式
    # c0 = np.polyfit(x[i - delta:i + delta], y[i - delta:i + delta], 4)  # 挠曲线系数
    # c1 = np.array([c0[0] * 4, c0[1] * 3, c0[2]*2])  # 1阶导函数系数
    # c2 = np.array([c1[0] * 3, c1[1]*2])  # 2阶导函数系数

    Y = np.poly1d(c0)
    ddy = np.poly1d(c2)
    dy = np.poly1d(c1)
    # y0[i] = Y(x[i])
    K[i] = ddy(x[i]) / ((1 + dy(x[i]) ** 2) ** (3 / 2))

plt.figure(3)
s = np.array(range(length))
plt.plot(s, K)
plt.title('3次多项式拟合下半径为1的半圆曲率识别结果')
plt.savefig('semicircle_3.png',dpi=1080)
plt.show()
plt.close(3)

