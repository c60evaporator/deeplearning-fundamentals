# %% -logy(交差エントロピー誤差)の可視化
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)  # 入力のリスト
y = -np.log(x)  # 関数適用
plt.plot(x, y)
plt.xlabel('y')
plt.ylabel('-log(y)')
plt.xlim(-0.02, 1.02)
plt.ylim(-0.1, 5)
# %%
