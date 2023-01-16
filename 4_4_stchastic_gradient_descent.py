# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import copy
from common.loss_funcions import cross_entropy_error, squared_error
from common.forward_functions import forward_middle, forward_last_classification
import seaborn as sns
import matplotlib.pyplot as plt

#numerical_gradient(cross_entropy_error, )

#sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット
#plt.xlabel('height [cm]')  # x軸のラベル
#plt.ylabel('weight [kg]')  # y軸のラベル

# %%
class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, n_layers, loss_type,
                 weight_init_std=0.01):
        """
        パラメータ等を初期化
        """
        self.params={'W': [],
                     'b': []}
        self.params['W'].append(weight_init_std * \
                            np.random.randn(input_size, hidden_size))  # 1層目の重みパラメータ
        for l in range(n_layers-1):
            self.params['b'].append(np.zeros(hidden_size))  # l層目のバイアスパラメータ
            self.params['W'].append(weight_init_std * \
                            np.random.randn(hidden_size, hidden_size)) # l+1層目の重みパラメータ
        self.params['b'].append(np.zeros(output_size))  # 最終層のバイアスパラメータ
        # 損失関数の種類
        self.loss_type = loss_type
        # 層数
        self.n_layers = n_layers
        # TODO: XとTもメンバ変数に含める＆ランダムサンプリング＆パラメータの更新もメソッド化＆fitメソッド作成
    

    def predict(self, X):
        """
        順伝播を全て計算
        """
        W1, W2, W3 = self.params['W'][0], self.params['W'][1], self.params['W'][2]
        b1, b2, b3 = self.params['b'][0], self.params['b'][1], self.params['b'][2]
        Z1 = forward_middle(X, W1, b1)
        Z2 = forward_middle(Z1, W2, b2)
        Z3 = forward_last_classification(Z2, W3, b3)
        return Z3

    def accuracy(self, X, T):
        """
        正解率Accuracyを計算
        """
        Y = self.predict(X)
        Y = np.argmax(Y, axis=1)  # 予測クラス (One-hotをクラスのインデックスに変換)
        T = np.argmax(T, axis=1)  # 正解クラス (One-hotをクラスのインデックスに変換)
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def loss(self, X, T):
        """
        損失関数の計算
        """
        Y = self.predict(X)
        if self.loss_type == 'cross_entropy':
            return cross_entropy_error(Y, T)
        elif self.loss_type == 'squared_error':
            return squared_error(Y, T)
        else:
            raise Exception('The `loss_type` argument should be "cross_entropy" or "squared_error"')
    
    def numerical_gradient(self, X, T, param_name, l):
        """
        層ごとの勾配を計算

        Parameters
        ----------
        X : numpy.ndarray
            入力データ (ランダムサンプリング後)
        T : numpy.ndarray
            正解データ (ランダムサンプリング後)
        param_name : {'W', 'b'}
            パラメータの種類 ('W': 重みパラメータ, 'b': バイアス)
        l : int
            何層目のパラメータか
        """
        # 勾配計算対象のパラメータを抽出 (self.params更新時に一緒に更新されないようDeepCopyする)
        P = copy.deepcopy(self.params[param_name][l])

        h = 1e-4  # 原書に記載された適切な微小変化量hの値として1e-4を採用
        P_ravel = np.ravel(P)  # Pが行列(重みパラメータ)の時、一旦ベクトルとして展開
        grad = np.zeros_like(P_ravel)  # Pと同じ形状のベクトルor行列を生成

        # パラメータごとに偏微分を計算
        for idx in range(P_ravel.size):
            # 微小変化量に相当するベクトル(該当パラメータに相当する成分のみh、他成分は全て0)
            h_vector = np.eye(P_ravel.size)[idx] * h
            # f(x+h)の計算
            P1 = (P_ravel + h_vector).reshape(P.shape)  # 微小変換させたP
            self.params[param_name][l] = P1  # 微小変化させたPをパラメータに反映
            fxh1 = self.loss(X, T)  # 微小変化後の損失関数を計算
            # f(x-h)の計算
            P2 = (P_ravel - h_vector).reshape(P.shape)  # 微小変換させたP
            self.params[param_name][l] = P2  # 微小変化させたPをパラメータに反映
            fxh2 = self.loss(X, T)  # 微小変化後の損失関数を計算
            # 偏微分の計算
            grad[idx] = (fxh1 - fxh2) / (2*h)
            # 微小変化させたPを元に戻す
            self.params[param_name][l] = P
        
        return grad.reshape(P.shape)

    def numerical_gradient_all(self, X, T):
        """
        全パラメータの勾配を計算
        """
        grads = {'W': [],
                 'b': []}
        for l in range(self.n_layers):
            grads['W'].append(self.numerical_gradient(X, T, 'W', l))
            grads['b'].append(self.numerical_gradient(X, T, 'b', l))


iris = sns.load_dataset("iris")
iris = iris[iris['species'].isin(['versicolor', 'virginica'])]

# 説明変数
X = iris[['petal_width', 'petal_length', 'sepal_width']].to_numpy()
input_size = X.shape[1]  # 説明変数の次元数
# 目的変数をOne-hot encoding
T = iris[['species']].to_numpy()
T = OneHotEncoder().fit_transform(T).toarray()
output_size = T.shape[1]  # 出力層のニューロン数 (クラス数)
# 学習データとテストデータ分割
X_train, X_test, T_train, T_test = train_test_split(X, T, shuffle=True, random_state=42)
train_size = X_train.shape[0]

# ハイパラ
iters_num = 100  # 学習(SGD)の繰り返し数
hidden_size = 2  # 隠れ層のニューロン数
n_layers = 3  # 層数
batch_size = 30  # バッチサイズ (サンプリング数)
learning_rate = 0.1  # 学習率
weight_init_std=0.01

# ニューラルネットワーク計算用クラス
network = ThreeLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers, 
                        loss_type='cross_entropy', weight_init_std=weight_init_std)

train_loss_list = []
for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    T_batch = T_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient_all(X_batch, T_batch)
    # パラメータの更新
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    # 学習経過の記録
    loss = network.loss(X_batch, T_batch)
    train_loss_list.append(loss)