# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.backprop_neuralnet import BackpropNeuralNet
import time

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
n_iter = 4000  # 学習(SGD)の繰り返し数
hidden_size = 2  # 隠れ層のニューロン数
n_layers = 3  # 層数
batch_size = 50  # バッチサイズ (サンプリング数)
learning_rate = 1.0  # 学習率
weight_init_std=0.1  # 重み初期値生成時の標準偏差

# ニューラルネットワーク計算用クラス (誤差逆伝播法バージョン)
network = BackpropNeuralNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                       learning_rate=learning_rate, batch_size=batch_size, n_iter=n_iter, 
                       loss_type='cross_entropy', activation_function='sigmoid',
                       weight_init_std=weight_init_std)
start = time.time()  # 時間計測用
# SGDによる学習
network.fit(X_train, T_train)
print(f'Training time={time.time() - start}sec')  # 学習時間を表示
# 精度評価
print(f'Accuracy={network.accuracy(X_test, T_test)}')
# 学習履歴のプロット
plt.plot(range(n_iter), network.train_loss_list)
plt.show()

# %% seaborn-analyzerで決定境界プロット
from seaborn_analyzer import classplot

# ニューラルネットワーク計算用クラス (誤差逆伝播法バージョン)
network = BackpropNeuralNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                       learning_rate=learning_rate, batch_size=batch_size, n_iter=n_iter, 
                       loss_type='cross_entropy', activation_function='sigmoid',
                       weight_init_std=weight_init_std)
# 決定境界をプロット
classplot.class_separator_plot(network, x=X_train, y=T_train, 
                               x_colnames=['petal_width', 'petal_length', 'sepal_width'])

# %%
