import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
matplotlib.use(backend='Qt5Agg')

img_path = './image_png/DSC00321.png'  # 可更换图像
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
ret, BW = cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY)

# 提取边缘
# contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
px = np.array(contours[0][:, :, 0]).squeeze()  # W
py = np.array(contours[0][:, :, 1]).squeeze()  # H
# 提取比例因子（实际距离和图像距离的比例）
factor = 2/(max(px) - min(px))

df = pd.DataFrame({'x': px, 'y': py})
# df = df[df['y']>=320]
df = df.groupby(['x']).max()
df.index.name = 'index'
df['x'] = df.index
df['z'] = df['y'].rolling(500, axis=0, min_periods=None, center=True).mean()
df['y'] = df['z']
df = df.dropna(axis=0, how='any')

df.sort_values('x', ascending=True, inplace=True)
px = np.array(df['x'])
py = np.array(df['y'])#.astype(np.int32)
print("最大像素点为{}，最小像素点为{}，共有像素点为{}".format(max(px),min(px),len(px)))

# # 修正后边缘提取效果
# t = np.array(range(len(px)))
# plt.figure()
# plt.plot(t, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.plot(t, px, color='red', marker='o',  linewidth=0.5, markersize=5)
# plt.legend(['像素点y方向变化','像素点x方向变化'])
# plt.show()

# # 修正后边缘在原图的效果
# plt.figure()
# origin = cv2.imread('./images/DSC00311.JPG')
# plt.imshow(origin)
# plt.plot(px, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()

# 图像距离和实际距离换算
px = px * factor
py = py * factor

# 间隔点计算转角
delta = 50  # 间隔点的间距像素，可以调节大小试试
corner = np.zeros(len(px)-delta)
for i in range(len(px)-delta):
    slope = (py[i]-py[i+delta])/(px[i]-px[i+delta])
    corner[i] = np.arctan(slope)
plt.figure(figsize=(16, 9))
plt.plot(range(len(corner)),corner, color='blue', marker='o',  linewidth=0.5, markersize=5)
plt.title('转角')
plt.show()
plt.close()

# 计算曲率
delta = 500  # 间隔点的间距像素，可以调节大小试试 
K = np.zeros(len(px))
for i in range(delta,len(px)-delta):
    n1 = (px[i]-px[i-delta],py[i]-py[i-delta])
    n2 = (-px[i]+px[i+delta],-py[i]+py[i+delta])
    L = np.sqrt((px[i+delta]-px[i-delta])**2 + (py[i+delta]-py[i-delta])**2)
    H = np.abs(n1[0]*n2[1]-n1[1]*n2[0])/L
    K[i] = 2 * H / (H**2 + (L/2)**2)

K = K[delta:len(px)-delta]
# print('曲率误差为{}和{}'.format(((max(K)-0.31)/0.31),((min(K)-0.31)/0.31)))
print('曲率误差为{}和{}'.format(((max(K)-1)/1),((min(K)-1)/1)))
t = np.array(range(len(K)))
plt.figure(figsize=(16, 9))
plt.plot(t, K, color='blue', marker='o',  linewidth=0.5, markersize=5)
plt.title('曲率')
plt.show()
