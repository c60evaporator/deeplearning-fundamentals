# %% ステップ関数の可視化
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

x = np.linspace(-10, 10, 200)  # 入力のリスト
y = np.vectorize(step_function)(x)  # 関数適用
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)

# %% シグモイド関数の可視化
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 200)  # 入力のリスト
y = np.vectorize(sigmoid)(x)  # 関数適用
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)

# %% ReLU関数の可視化
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 200)  # 入力のリスト
y = np.vectorize(relu)(x)  # 関数適用
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-1, 11)

# %%
