import numpy as np
from PIL import Image
import glob
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
matplotlib.use(backend='Qt5Agg')
plt.rcParams['font.sans-serif']=['Microsoft YaHei']


def edge_func(img_path, origin_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, BW = cv2.threshold(gray, 120, 1, cv2.THRESH_BINARY)

    (H, W) = BW.shape[:2]
    area = H * W  # 图片面积
    discard = 0.001  # 丢弃面积率
    BW = morphology.remove_small_objects(BW > 0, discard * area, connectivity=2)
    BW = BW.astype(np.uint8)

    # 提取边缘
    y, x = np.where(BW == 1)
    df = pd.DataFrame({'x': x, 'y': y})
    data_group = df.groupby(['x']).max()
    data_group.index.name = 'index'
    data_group['x'] = data_group.index
    data_group.sort_values('x', ascending=True, inplace=True)
    x = np.array(data_group['x']).astype(np.int)
    y = np.array(data_group['y']).astype(np.int)

    plt.figure(3)
    origin = cv2.imread(origin_path)
    plt.imshow(origin)
    plt.plot(x, y, color='blue', marker='o', linewidth=0.5, markersize=5)
    plt.show()
    plt.close(3)

    # 提取比例因子
    factor = 2 / (max(x) - min(x))

    # 计算曲率
    x = x * factor
    y = y * factor

    length = len(x)
    delta = 500
    K = np.zeros(length)

    # 全部3274个像素，取局部600个像素
    for i in range(delta, length - delta):
        # 3次多项式
        c0 = np.polyfit(x[i - delta:i + delta], y[i - delta:i + delta], 3)  # 挠曲线系数
        c1 = np.array([c0[0] * 3, c0[1] * 2, c0[2]])  # 1阶导函数系数
        c2 = np.array([c1[0] * 2, c1[1]])  # 2阶导函数系数

        Y = np.poly1d(c0)
        ddy = np.poly1d(c2)
        dy = np.poly1d(c1)
        # y0[i] = Y(x[i])
        K[i] = ddy(x[i]) / ((1 + dy(x[i]) ** 2) ** (3 / 2))

    plt.figure(4)
    s = np.array(range(length))
    plt.plot(s, K)
    plt.title('3次多项式拟合下半径为1的半圆曲率识别结果')
    # plt.savefig('semicircle_3.png', dpi=1080)
    plt.show()
    plt.close(4)


def main():
    img_path = './image_png/DSC00313.png'
    origin_path = './images/DSC00313.JPG'
    edge_func(img_path, origin_path)


if __name__=='__main__':
    main()
