#%%
import numpy as np
import matplotlib.pyplot as plt
from common.activation_functions import sigmoid, relu

input_data = np.random.randn(1000, 100)  # 1000個の入力データ(標準正規分布から生成)
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層

# 活性化関数を変えて実験する
for activation_function in ['sigmoid', 'relu']:

    # 重み初期値生成時の標準偏差を変えて実験する
    weight_init_stds = {'Std=1.0': 1.0,
                        'Std=0.01': 0.01,
                        'Xavier': np.sqrt(1.0 / node_num), 
                        'He': np.sqrt(2.0 / node_num)}
    
    # 結果描画用のキャンバス
    fig, axes = plt.subplots(4, hidden_layer_size, figsize=(3*hidden_layer_size, 4*4))
    fig.suptitle(f'activation_function = {activation_function}', fontsize=32)

    for i_std, (std_label, weight_init_std) in enumerate(weight_init_stds.items()):
        activations = {}  # 各層出力格納用
        # 入力データ
        x = input_data
        # 全層ループ
        for i in range(hidden_layer_size):
            # 前層出力を入力とする
            if i != 0:
                x = activations[i-1]

            # 重みパラメータの初期化 (標準偏差weight_init_stdの正規分布)
            w = np.random.randn(node_num, node_num) * weight_init_std
            # Affineレイヤ
            a = np.dot(x, w)
            # 活性化関数(Sigmoid)
            if activation_function == 'sigmoid':
                z = sigmoid(a)
            # 活性化関数 (ReLU)
            elif activation_function == 'relu':
                z = relu(a)
            # 各層の出力を格納
            activations[i] = z

        # 層ごとにヒストグラムを描画
        for i, a in activations.items():
            ax = axes[i_std][i]
            ax.set_title(str(i+1) + "-layer")
            if i == 0:
                ax.set_ylabel(std_label, fontsize=18)
            else:
                ax.set_yticks([])
            ax.hist(a.flatten(), 30, range=(0,1))
    plt.show()

# %% 比較
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.backprop_advanced import BackpropAdvancedNet

# データ読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'].isin(['versicolor', 'virginica'])]  # 'versicolor', 'virginica'の2クラスに絞る

# 説明変数
X = iris[['petal_width', 'petal_length', 'sepal_width']].to_numpy()
# 目的変数をOne-hot encoding
T = iris['species'].to_numpy()

# 学習データとテストデータ分割
X_train, X_test, T_train, T_test = train_test_split(X, T, shuffle=True, random_state=42)
train_size = X_train.shape[0]
# ハイパーパラメータ
n_iter = 5000  # 学習(SGD)の繰り返し数
hidden_size = 10  # 隠れ層のニューロン数
n_layers = 5  # 層数
batch_size = 50  # バッチサイズ (サンプリング数)
learning_rate = {'sigmoid': 1.0, 'relu': 0.05}  # 学習率
weight_init_std='auto'  # 重み初期値生成時の標準偏差

# 活性化関数を変えて実験する
for activation_function in ['sigmoid', 'relu']:

    # 重み初期値生成時の標準偏差を変えて実験する
    weight_init_stds = {'Std=1.0': 1.0,
                        'Std=0.01': 0.01,
                        'Auto': 'auto'}
    
    # 結果描画用のキャンバス
    fig, axes = plt.subplots(3, 1, figsize=(6, 3*4))
    fig.suptitle(f'activation_function = {activation_function}', fontsize=32)

    for i_std, (std_label, weight_init_std) in enumerate(weight_init_stds.items()):

        # ニューラルネットワーク計算用クラス (誤差逆伝播法バージョン)
        network = BackpropAdvancedNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                            learning_rate=learning_rate[activation_function], batch_size=batch_size, n_iter=n_iter, 
                            loss_type='cross_entropy', activation_function=activation_function,
                            weight_init_std=weight_init_std)
        # SGDによる学習
        network.fit(X_train, T_train)

        # 学習履歴のプロット
        ax = axes[i_std]
        ax.set_ylabel(std_label, fontsize=18)
        ax.set_ylim(-0.1, 2.0)
        ax.plot(range(n_iter), network.train_loss_list)
plt.show()
# %%
