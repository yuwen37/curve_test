import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
matplotlib.use(backend='Qt5Agg')

img_path = './image_png/DSC00309.png'  # 1200万像素
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
ret, BW = cv2.threshold(gray, 100, 1, cv2.THRESH_BINARY)
# BW = ndimage.binary_fill_holes(BW).astype(np.uint8)
# (H, W) = BW.shape[:2]
# area = H * W  # 图片面积
# discard = 0.001  # 丢弃面积率
# BW = morphology.remove_small_objects(BW > 0, discard*area, connectivity=2)
# BW = BW.astype(np.uint8)

# # 提取边缘方法一
# y, x = np.where(BW == 1)
# df = pd.DataFrame({'x': x, 'y': y})
# data_group = df.groupby(['x']).max()
# data_group.index.name = 'index'
# data_group['x'] = data_group.index
# data_group.sort_values('x', ascending=True, inplace=True)
# x = np.array(data_group['x']).astype(np.int)
# y = np.array(data_group['y']).astype(np.int)

# 提取边缘方法二
# contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
px = np.array(contours[0][:, :, 0]).squeeze()  # W
py = np.array(contours[0][:, :, 1]).squeeze()  # H
# 提取比例因子
factor = 2/(max(px) - min(px))

df = pd.DataFrame({'x': px, 'y': py})
df = df[df['y']>=320]
df = df.groupby(['x']).max()
df.index.name = 'index'
df['x'] = df.index
df['z'] = df['y'].rolling(200, axis=0, min_periods=None, center=True).mean()
df['y'] = df['z']
df = df.dropna(axis=0, how='any')

# 消除台阶
# platform = np.array(df['y'].iloc[1:]) - np.array(df['y'].iloc[:-1])
# index = np.where(platform == 0)[0]
# df['z'] = df['x'].rolling(2, axis=0).mean()
# df['x'].iloc[index+1] = np.array(df['z'].iloc[index+1])
# df['x'].iloc[index] = np.array(df['z'].iloc[index+1])
# df.drop(['z'], axis=1, inplace=True)
# df.drop_duplicates(subset=None, keep='first', inplace=True)

df.sort_values('x', ascending=True, inplace=True)
# x = np.array(df['x']).astype(np.int)
# y = np.array(df['y']).astype(np.int)
px = np.array(df['x'])
py = np.array(df['y'])

# 展示边缘提取效果
# plt.figure(3)
# origin = cv2.imread('./images/DSC00322.JPG')
# plt.imshow(origin)
# plt.plot(px, py, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()

print("最大像素点为{}，最小像素点为{}，共有像素点为{}".format(max(px),min(px),len(px)))


# 图像距离和实际距离换算
px = px * factor
py = py * factor
# plt.figure(2)
# plt.plot(x, y)
# plt.show()
# plt.close(2)


# ## 三次贝塞尔曲线
# # 初始化控制点坐标及曲率
# c1_x = np.zeros(len(px) - 1)
# c1_y = np.zeros(len(py) - 1)
# c2_x = np.zeros(len(px) - 1)
# c2_y = np.zeros(len(py) - 1)
# K = np.zeros(len(py) - 1)
# # 计算每个端点的曲率
# for i, p in enumerate(px[:-1]):  # p代表端点
#     if i == 0:
#         c1_x[0] = (2 * px[0] + px[1]) / 3
#         c1_y[0] = (2 * py[0] + py[1]) / 3
#         c2_x[0] = (px[0] + 2 * px[1]) / 3
#         c2_y[0] = (py[0] + 2 * py[1]) / 3
#     else:
#         c1_x[i] = 2 * px[i] - c2_x[i - 1]
#         c1_y[i] = 2 * py[i] - c2_y[i - 1]
#         c2_x[i] = c1_x[i - 1] - 2 * c2_x[i - 1] + 2 * c1_x[i]
#         c2_y[i] = c1_y[i - 1] - 2 * c2_y[i - 1] + 2 * c1_y[i]
#     if (c1_x[i]<px[i] or c1_x[i]>px[i+1] or c2_x[i]<px[i] or c2_x[i]>px[i+1]):
#         c1_x[i] = (2 * px[i] + px[i+1]) / 3
#         c1_y[i] = (2 * py[i] + py[i+1]) / 3
#         c2_x[i] = (px[i] + 2 * px[i+1]) / 3
#         c2_y[i] = (py[i] + 2 * py[i+1]) / 3
#     t = 0.5
#     dx = 3*(1-t)**2*(c1_x[i]-px[i])+6*(1-t)*t*(c2_x[i]-c1_x[i])+3*t**2*(px[i+1]-c2_x[i])
#     dy = 3*(1-t)**2*(c1_y[i]-py[i])+6*(1-t)*t*(c2_y[i]-c1_y[i])+3*t**2*(py[i+1]-c2_y[i])
#     ddx = -6*(1-t)*(c1_x[i]-px[i])+6*(1-2*t)*(c2_x[i]-c1_x[i])+6*t*(px[i+1]-c2_x[i])
#     ddy = -6*(1-t)*(c1_y[i]-py[i])+6*(1-2*t)*(c2_y[i]-c1_y[i])+6*t*(py[i+1]-c2_y[i])
#     K[i] = np.abs(dx*ddy-dy*ddx)/(np.abs(dx**2+dy**2)**(3/2))
# plt.figure(4)
# s = np.array(range(len(K)))
# plt.plot(s, K)
# plt.show()
# plt.close(4)

# ## 2次贝塞尔曲线（能画出曲线了，但是端点连接效果很差）
# # 初始化控制点坐标及曲率
# c_x = np.zeros(len(px) - 1)
# c_y = np.zeros(len(py) - 1)
# K = np.zeros(len(py) - 1)
# bezier_x =np.zeros((len(py) - 1)*10)
# bezier_y =np.zeros((len(py) - 1)*10)
# # 计算每个端点的曲率
# for i, p in enumerate(px[:-1]):  # p代表端点
#     if i == 0:
#         c_x[0] = (px[0] + px[1]) / 2
#         c_y[0] = (py[0] + py[1]) / 2
#     else:
#         c_x[i] = 2 * px[i] - c_x[i - 1]
#         c_y[i] = 2 * py[i] - c_y[i - 1]
#     if (c_x[i]<px[i] or c_x[i]>px[i+1]):
#         c_x[i] = (px[i] + px[i+1]) / 2
#         c_y[i] = (py[i] + py[i+1]) / 2
#     t = 0
#     for loc_t in range(10):
#         bezier_x[10*i+loc_t] = (1-loc_t*0.1)**2*px[i]+2*(1-loc_t*0.1)*(loc_t*0.1)*c_x[i]+(loc_t*0.1)**2*px[i+1]
#         bezier_y[10*i+loc_t] = (1-loc_t*0.1)**2*py[i]+2*(1-loc_t*0.1)*(loc_t*0.1)*c_y[i]+(loc_t*0.1)**2*py[i+1]
#     dx = -2*(1-t)*px[i]+2*(1-2*t)*c_x[i]+2*t*px[i+1]
#     dy = -2*(1-t)*py[i]+2*(1-2*t)*c_y[i]+2*t*py[i+1]
#     ddx = 2*px[i]-4*c_x[i]+2*px[i+1]
#     ddy = 2*py[i]-4*c_y[i]+2*py[i+1]
#     K[i] = np.abs(dx * ddy - dy * ddx) / (np.abs(dx ** 2 + dy ** 2) ** (3 / 2))
# plt.figure(4)
# s = np.array(range(len(K)))
# plt.plot(s, K, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()
# plt.close(4)
# # 展示边缘提取效果
# plt.figure(5)
# origin = cv2.imread('./images/DSC00322.JPG')
# plt.imshow(origin)
# plt.plot(px/factor, py/factor, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.plot(bezier_x/factor, bezier_y/factor, color='red', marker='o',  linewidth=0.5, markersize=3)
# plt.show()
# plt.close(5)

# ## 2次贝塞尔曲线_提升版（加入了权重）（有点不对劲）
# n = len(px) if len(px)%2 == 1 else len(px)-1
# w = np.ones(n-1)
# K = np.zeros(n-1)
# bezier_x = np.zeros(n*10)
# bezier_y = np.zeros(n*10)
# for i in range(n//2):
# #     if i == 0:
# #         w[:2] = 1
# #     else:
# #         e = px[2 * i - 1] * w[2 * i - 1] - px[2 * i]
# #         f = py[2 * i - 1] * w[2 * i - 1] - py[2 * i]
# #         ad_bc = -px[2 * i] * py[2 * i + 1] + px[2 * i + 1] * py[2 * i]
# #         w[2 * i] = (-py[2 * i + 1] * e + px[2 * i + 1] * f) / ad_bc
# #         w[2 * i + 1] = (px[2 * i] * f - py[2 * i] * e) / ad_bc
#     for loc_t in range(10):
#         bezier_x[10*i+loc_t] = (1-loc_t*0.1)**2*px[2*i]*w[2*i]+2*(1-loc_t*0.1)*(loc_t*0.1)*px[2*i+1]*w[2*i+1]+(loc_t*0.1)**2*px[2*i+2]
#         bezier_y[10*i+loc_t] = (1-loc_t*0.1)**2*py[2*i]*w[2*i]+2*(1-loc_t*0.1)*(loc_t*0.1)*py[2*i+1]*w[2*i+1]+(loc_t*0.1)**2*py[2*i+2]
#     t = 0
#     # bezier_x[i] = (1 - t) ** 2 * px[2 * i] * w[2 * i] + 2 * (1 - t) * (t) * \
#     #                            px[2 * i + 1] * w[2 * i + 1] + (t) ** 2 * px[2 * i + 2]
#     # bezier_y[i] = (1 - t) ** 2 * py[2 * i] * w[2 * i] + 2 * (1 - t) * (t) * \
#     #               py[2 * i + 1] * w[2 * i + 1] + (t) ** 2 * py[2 * i + 2]
#     dx = (-2*(1-t)*px[2*i]*w[2*i] +
#           2*(1-2*t)*px[2*i+1]*w[2*i+1] +
#           2*t*px[2*i+2])
#     dy = (-2*(1-t)*py[2*i]*w[2*i] +
#           2*(1-2*t)*py[2*i+1]*w[2*i+1] +
#           2*t*py[2*i+2])
#     ddx = 2 * px[2*i]*w[2*i] - 4*px[2*i+1]*w[2*i+1] + 2*px[2*i+2]
#     ddy = 2 * py[2*i]*w[2*i] - 4*py[2*i+1]*w[2*i+1] + 2*py[2*i+2]
#     K[i] = np.abs(dx * ddy - dy * ddx) / (np.abs(dx ** 2 + dy ** 2) ** (3 / 2))
# plt.figure(4)
# s = np.array(range(len(K)))
# plt.plot(s, K, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()
# plt.close(4)
# # 展示边缘提取效果
# plt.figure(5)
# origin = cv2.imread('./images/DSC00322.JPG')
# plt.imshow(origin)
# plt.plot(px/factor, py/factor, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.plot(bezier_x/factor, bezier_y/factor, color='red', marker='o',  linewidth=0.5, markersize=7)
# plt.show()
# plt.close(5)

# ## 3次贝塞尔曲线_提升版（加入了权重）
# n = (len(px)//3)-1 if len(px)%3 == 0 else len(px)//3
# w = np.ones(n*2)
# K = np.zeros(n)
# bezier_x = np.zeros(n*10)
# bezier_y = np.zeros(n*10)
# # 计算权重
# # for i in range(1,n-1):
# #     a = px[3*i+1]
# #     b = -px[3*i-1]
# #     c = py[3*i+1]
# #     d = -py[3*i-1]
# #     e = 2*px[3*i]
# #     f = 2*py[3*i]
# #     w[2 * i] = (d * e - b * f) / (a * d - b * c)
# #     w[2 * i - 1] = (a * f - c * e) / (a * d - b * c)
# for i in range(n):
#     for loc_t in range(10):
#         bezier_x[10 * i + loc_t] = ((1 - loc_t * 0.1) ** 3 * px[3 * i] +
#                                     3 * (1 - loc_t * 0.1)**2 * (loc_t * 0.1) * px[3 * i + 1] * w[2 * i] +
#                                     3 * (1 - loc_t * 0.1) * (loc_t * 0.1) ** 2 * px[3 * i + 2] * w[2 * i + 1] +
#                                     (loc_t * 0.1) ** 3 * px[3 * i + 3])
#         bezier_y[10 * i + loc_t] = ((1 - loc_t * 0.1) ** 3 * py[3 * i] +
#                                     3 * (1 - loc_t * 0.1)**2 * (loc_t * 0.1) * py[3 * i + 1] * w[2 * i] +
#                                     3 * (1 - loc_t * 0.1) * (loc_t * 0.1) ** 2 * py[3 * i + 2] * w[2 * i + 1] +
#                                     (loc_t * 0.1) ** 3 * py[3 * i + 3])
#     t = 0
#     dx = (-3*(1-t)**2*px[3*i] +
#           3*(1-3*t)*(1-t)*px[3*i+1]*w[2*i] +
#           3*(2*t-3*t**2)*px[3*i+2]*w[2*i+1] + 
#           3*t**2*px[3*i+3])
#     dy = (-3*(1-t)**2*py[3*i] +
#           3*(1-3*t)*(1-t)*py[3*i+1]*w[2*i] +
#           3*(2*t-3*t**2)*py[3*i+2]*w[2*i+1] + 
#           3*t**2*py[3*i+3])
#     ddx = (6*(1-t)*px[3*i] +
#           3*(6*t-4)*px[3*i+1]*w[2*i] +
#           3*(2-6*t)*px[3*i+2]*w[2*i+1] + 
#           6*t*px[3*i+3])
#     ddy = (6*(1-t)*py[3*i] +
#           3*(6*t-4)*py[3*i+1]*w[2*i] +
#           3*(2-6*t)*py[3*i+2]*w[2*i+1] + 
#           6*t*py[3*i+3])
#     K[i] = np.abs(dx * ddy - dy * ddx) / (np.abs(dx ** 2 + dy ** 2) ** (3 / 2))
# plt.figure(4)
# s = np.array(range(len(K)))
# plt.plot(s, K, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.show()
# plt.close(4)
# # 展示边缘提取效果
# plt.figure(5)
# origin = cv2.imread('./images/DSC00322.JPG')
# plt.imshow(origin)
# plt.plot(px/factor, py/factor, color='blue', marker='o',  linewidth=0.5, markersize=5)
# plt.plot(bezier_x/factor, bezier_y/factor, color='red', marker='o',  linewidth=0.5, markersize=3)
# plt.show()
# plt.close(5)


# 普通多项式拟合
length = len(px)
delta = 500
K = np.zeros(length)

# 全部3274个像素，取局部600个像素
for i in range(delta, length-delta):
    # 2次多项式
    # c0 = np.polyfit(px[i - delta:i + delta], py[i - delta:i + delta], 2)
    # c1 = np.array([c0[0]*2, c0[1]*1])
    # c2 = np.array([c1[1]])

    # 3次多项式
    c0 = np.polyfit(px[i - delta:i + delta], py[i - delta:i + delta], 3)  # 挠曲线系数
    c1 = np.array([c0[0] * 3, c0[1] * 2, c0[2]])  # 1阶导函数系数
    c2 = np.array([c1[0] * 2, c1[1]])  # 2阶导函数系数

    # 4次多项式
    # c0 = np.polyfit(px[i - delta:i + delta], py[i - delta:i + delta], 4)  # 挠曲线系数
    # c1 = np.array([c0[0] * 4, c0[1] * 3, c0[2]*2])  # 1阶导函数系数
    # c2 = np.array([c1[0] * 3, c1[1]*2])  # 2阶导函数系数

    Y = np.poly1d(c0)
    ddy = np.poly1d(c2)
    dy = np.poly1d(c1)
    # y0[i] = Y(x[i])
    K[i] = ddy(px[i]) / ((1 + dy(px[i]) ** 2) ** (3 / 2))

plt.figure(4)
s = np.array(range(length))
plt.plot(s, K)
plt.title('3次多项式拟合下半径为1的半圆曲率识别结果')
# plt.savefig('semicircle_3.png', dpi=1080)
plt.show()
plt.close(4)

plt.figure(1)
origin = cv2.imread('./images/DSC00309.JPG')
plt.imshow(origin)
plt.plot(px/factor, py/factor, color='blue', marker='o',  linewidth=0.5, markersize=5)
plt.show()
plt.close(1)