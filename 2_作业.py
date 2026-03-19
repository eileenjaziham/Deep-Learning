import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x*w +b

def loss(x,y):
    y_pred = forward(x)
    return (y-y_pred)*(y-y_pred)

w_range = np.arange(0.0,4.1,0.1)
b_range = np.arange(-2.0,+2.0,0.1)

W, B = np.meshgrid(w_range, b_range) 
#meshgrid 函数将两个一维数组转换为二维坐标矩阵
J = np.zeros_like(W, dtype=np.float64) 
# np.zeros_like(W) 创建一个与 W 形状完全相同的新数组
# 所有元素初始化为 0
# dtype=np.float64 指定数据类型为双精度浮点数，确保计算精度
#相当于先创建了x，y轴，再复制成了z轴

for i in range(W.shape[0]):        # 遍历 b，对于二维数组，shape 返回元组 (行数, 列数)
    for j in range(W.shape[1]):    # 遍历 w
        w = W[i, j]
        b = B[i, j]
        l_sum = 0.0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        J[i, j] = l_sum / len(x_data)    # MSE

fig = plt.figure() #创建一个新的画布
ax = fig.add_subplot(111, projection="3d")
#在画布中添加一个子画布，1行1列中的第一个子画布
#而且是三维的

ax.plot_surface(W, B, J, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
# 绘制三维曲面图
# W：w 的网格坐标（x 轴）
# B：b 的网格坐标（y 轴）
# J：对应 (w, b) 的损失值（z 轴）
# cmap="viridis"：颜色映射方案
# linewidth=0：曲面网格线宽为 0（更平滑）
# antialiased=True：抗锯齿，使曲面更平滑
# alpha=0.95：透明度（1 为不透明）
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("MSE Loss J(w,b)")
ax.set_title("Cost Surface for y = wx + b")

plt.show()

