# %% SGD
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.backprop_advanced import BackpropAdvancedNet
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
hidden_size = 10  # 隠れ層のニューロン数
n_layers = 5  # 層数
batch_size = 50  # バッチサイズ (サンプリング数)
weight_init_std='auto'  # 重み初期値生成時の標準偏差 (Xavierの初期値を使用)
learning_rate = {'sgd': 1.0, 'momentum': 0.5, 'adagrad': 0.1, 'rmsprop': 0.01, 'adam': 0.01}  # 学習率
momentum = {'sgd': None, 'momentum': 0.5, 'adagrad': None, 'rmsprop': None, 'adam': None}# 勾配移動平均の減衰率ハイパーパラメータ(モーメンタム)
beta_1 = {'sgd': None, 'momentum': None, 'adagrad': None, 'rmsprop': None, 'adam': 0.5}# 勾配移動平均の減衰率ハイパーパラメータ(Adam)
beta_2 = {'sgd': None, 'momentum': None, 'adagrad': None, 'rmsprop': 0.999, 'adam': 0.999}# 勾配2乗和の減衰率ハイパーパラメータ(RMSprop, Adam)
epsilon = {'sgd': None, 'momentum': None, 'adagrad': 1e-8, 'rmsprop': 1e-8, 'adam': 1e-8}# ゼロ除算によるエラーを防ぐハイパーパラメータ(AdaGrad, RMSprop, Adam)

# 最適化手法を変えてプロット
for algo in ['sgd', 'momentum' ,'adagrad', 'rmsprop', 'adam']:

    # ニューラルネットワーク計算用クラス (最適化アルゴリズム修正)
    network = BackpropAdvancedNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                        batch_size=batch_size, n_iter=n_iter, 
                        loss_type='cross_entropy', activation_function='sigmoid', weight_init_std=weight_init_std,
                        solver=algo, 
                        learning_rate=learning_rate[algo],
                        momentum=momentum[algo],
                        beta_1=beta_1[algo],
                        beta_2=beta_2[algo],
                        epsilon=epsilon[algo]
                        )
    start = time.time()  # 時間計測用
    # SGDによる学習
    network.fit(X_train, T_train)
    print(f'Training time={time.time() - start}sec')  # 学習時間を表示
    # 精度評価
    print(f'Accuracy={network.accuracy(X_test, T_test)}')
    # 学習履歴のプロット
    plt.plot(range(n_iter), network.train_loss_list)
    plt.title(algo)
    plt.show()
# %%
