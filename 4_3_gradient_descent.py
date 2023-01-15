# %% 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# (x1,x2)格子点を作成
x1_grid = np.linspace(-3, 3, 13)
x2_grid = np.linspace(-3, 3, 13)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
# 損失関数f=x1^2+x2^2とする
def loss_func(x):
    return x[0]**2 + x[1]**2
# 格子点全てで損失関数を計算
F = np.zeros_like(X1)
for idx1 in range(x1_grid.size):
    for idx2 in range(x2_grid.size):
        F[idx1, idx2] = loss_func(np.array([X1[idx1, idx2], X2[idx1, idx2]]))

# 曲面をプロット
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("parameter1", size = 16)  # x1軸
ax.set_ylabel("parameter2", size = 16)  # x2軸
ax.set_zlabel("f", size = 16)  # f軸
ax.plot_surface(X1, X2, F, cmap = "YlGn_r")
plt.show()

# 勾配を計算する関数
def numerical_gradient(f, x):
    h = 1e-4  # 原書に記載された適切な微小変化量hの値として1e-4を採用
    grad = np.zeros_like(x)  # xと同じ形状のベクトルを生成
    # 変数ごとに偏微分を計算
    for idx in range(x.size):
        # f(x+h)の計算
        h_vector = np.eye(x.size)[idx] * h  # 微小変化量に相当するベクトル(該当変数のみh、他成分は全て0)
        fxh1 = f(x + h_vector)
        # f(x-h)の計算
        fxh2 = f(x - h_vector)
        # 偏微分の計算
        grad[idx] = (fxh1 - fxh2) / (2*h)
    return grad
# 格子ごとに勾配を計算
G1, G2 = np.zeros_like(X1), np.zeros_like(X2)
for idx1 in range(x1_grid.size):
    for idx2 in range(x2_grid.size):
        G1[idx1, idx2], G2[idx1, idx2] = numerical_gradient(loss_func, np.array([X1[idx1, idx2], X2[idx1, idx2]]))

# 勾配をプロット
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111)
ax.grid()  # 格子点を表示
for idx1 in range(x1_grid.size):  # x1軸を走査
    for idx2 in range(x2_grid.size):  # x2軸を走査
        ax.quiver(X1[idx1, idx2], X2[idx1, idx2], G1[idx1, idx2], G2[idx1, idx2], 
            color = "red", angles = 'xy', scale_units = 'xy', scale = 10)
ax.set_xlabel("parameter1", size = 16)  # x1軸
ax.set_ylabel("parameter2", size = 16)  # x2軸
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
plt.show()

# %%
