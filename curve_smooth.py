import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from scipy.fftpack import fft,ifft
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# matplotlib.use(backend='Qt5Agg')

img_path = './image_png/DSC00309.png'  # 1200万像素
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
ret, BW = cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY)

# 提取边缘
# contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
px = np.array(contours[0][:, :, 0]).squeeze()  # W
py = np.array(contours[0][:, :, 1]).squeeze()  # H
df = pd.DataFrame({'x': px, 'y': py})
df = df[df['y']>=320]
df = df.groupby(['x']).max()
df.index.name = 'index'
df['x'] = df.index
df.sort_values('x', ascending=True, inplace=True)
df['z'] = df['y'].rolling(50, axis=0, min_periods=None, center=True).mean()
df['y'] = df['z']
df = df.dropna(axis=0, how='any')
px = np.array(df['x'])
py = np.array(df['y'])
pixel = np.array(range(len(px)))
# # 边缘提取效果
# t = np.array(range(len(px)))
# plt.figure()
# plt.plot(pixel, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.plot(pixel, px, color='red', marker='o',  linewidth=0.5, markersize=5)
# plt.legend(['像素点y方向变化','像素点x方向变化'])
# plt.show()

# 傅里叶变换
# fft_y = fft(py)
# abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
# angle_y = np.angle(fft_y)  # 取复数的角度
# plt.figure()
# plt.plot(px, abs_y)
# plt.title('双边振幅谱（未归一化）')
# plt.show()

# test_y =fft_y
# for i in range(len(fft_y)):
#     if i > 0.45*len(fft_y): # or i >= 0.55 *len(fft_y)
#        test_y[i]=0
# test = np.fft.ifft(test_y)
# plt.figure(1)
# # plt.plot(px, test, color='blue', marker='o',  linewidth=0.5, markersize=5)
# # plt.plot(px, py, color='red', marker='o',  linewidth=0.5, markersize=3)
# plt.show()
# plt.close(1)

# print('x的像素点个数：{}，x的最大值：{}，x的最小值：{}'.format(len(px),max(px),min(px)))
# # 展示边缘提取效果
# plt.figure(2)
# origin = cv2.imread('./images/DSC00322.JPG')
# plt.imshow(origin)
# plt.plot(px, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()

# 修正后边缘提取效果
t = np.array(range(len(px)))
plt.figure()
plt.plot(t, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
plt.plot(t, px, color='red', marker='o',  linewidth=0.5, markersize=5)
plt.legend(['像素点y方向变化','像素点x方向变化'])
plt.show()

