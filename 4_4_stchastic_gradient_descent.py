# %%
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import copy
from common.loss_funcions import cross_entropy_error, squared_error
from common.forward_functions import forward_middle, forward_last_classification
import seaborn as sns
import matplotlib.pyplot as plt

class SGDNeuralNet:
    def __init__(self, X, T,
                 hidden_size, n_layers, loss_type,
                 weight_init_std=0.01):
        """
        パラメータ等を初期化

        Parameters
        ----------
        X : numpy.ndarray 2D
            入力データ
        T : numpy.ndarray 1D or 2D
            正解データ
        hidden_size : int
            隠れ層の1層あたりニューロン
        n_layers : int
            層数 (隠れ層の数 - 1)
        loss_type : {'cross_entropy', 'squared_error'}
            損失関数の種類 ('cross_entropy': 交差エントロピー誤差, 'squared_error': 2乗和誤差)
        weight_init_std : float
            初期パラメータ生成時の標準偏差
        """
        # 各種メンバ変数の入力
        input_size = X.shape[1]  # 説明変数の次元数(1層目の入力数)
        output_size = T.shape[1] if T.ndim == 2 else np.unique(T).size  # クラス数 (出力層のニューロン数)
        self.loss_type = loss_type  # 損失関数の種類
        self.n_layers = n_layers  # 層数
        # パラメータを初期化
        self.params={'W': [],
                     'b': []}
        self.params['W'].append(weight_init_std * \
                            np.random.randn(input_size, hidden_size))  # 1層目の重みパラメータ
        for l in range(n_layers-1):
            self.params['b'].append(np.zeros(hidden_size))  # l+1層目のバイアスパラメータ
            self.params['W'].append(weight_init_std * \
                            np.random.randn(hidden_size, hidden_size)) # l+2層目の重みパラメータ
        self.params['b'].append(np.zeros(output_size))  # 最終層のバイアスパラメータ

    def _one_hot_encoding(self, T):
        """
        One-hot encodingを実行する
        """
        # Tが1次元ベクトルなら2次元に変換してOne-hot encodingする
        if T.ndim == 1:
            T_onehot = T.reshape([T.size, 1])
            self.one_hot_encoder_ = OneHotEncoder().fit(T_onehot)  # エンコーダをメンバ変数として保持
            T_onehot = self.one_hot_encoder_.transform(T_onehot).toarray()
        # Tが2次元ベクトルなら既にOne-hot encodingされているとみなしてそのまま返す
        else:
            T_onehot = T
        return T_onehot

    def _one_hot_encoding_reverse(self, T):
        """
        One-hot encodingから元のカテゴリ変数に戻す
        """
        # One-hotをクラスのインデックスに変換
        T_label = np.argmax(T, axis=1)
        # メンバ変数として保持したエンコーダを参照にカテゴリ変数に戻す
        T_cat = np.vectorize(lambda x: self.one_hot_encoder_.categories_[0][x])(T_label)
        return T_cat
    
    def _predict_onehot(self, X):
        """
        順伝播を全て計算(One-hot encodingで出力)
        """
        W1, W2, W3 = self.params['W'][0], self.params['W'][1], self.params['W'][2]
        b1, b2, b3 = self.params['b'][0], self.params['b'][1], self.params['b'][2]
        Z1 = forward_middle(X, W1, b1)
        Z2 = forward_middle(Z1, W2, b2)
        Z3 = forward_last_classification(Z2, W3, b3)
        return Z3
    
    def predict(self, X):
        """
        順伝播を全て計算(クラス名で出力)
        """
        Y = self._predict_onehot(X)
        Y = self._one_hot_encoding_reverse(Y)
        return Y

    def select_minibatch(self, X, T):
        """
        ステップ1: ミニバッチの取得
        """
        batch_mask = np.random.choice(train_size, batch_size)  # ランダムサンプリング
        X_batch = X[batch_mask]
        T_batch = T[batch_mask]
        return X_batch, T_batch

    def _loss(self, X, T):
        """
        損失関数の計算
        """
        Y = self._predict_onehot(X)
        if self.loss_type == 'cross_entropy':
            return cross_entropy_error(Y, T)
        elif self.loss_type == 'squared_error':
            return squared_error(Y, T)
        else:
            raise Exception('The `loss_type` argument should be "cross_entropy" or "squared_error"')
    
    def _numerical_gradient(self, X, T, param_name, l):
        """
        層ごとの勾配を計算

        Parameters
        ----------
        X : numpy.ndarray 2D
            入力データ (ランダムサンプリング後)
        T : numpy.ndarray 2D
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
            fxh1 = self._loss(X, T)  # 微小変化後の損失関数を計算
            # f(x-h)の計算
            P2 = (P_ravel - h_vector).reshape(P.shape)  # 微小変換させたP
            self.params[param_name][l] = P2  # 微小変化させたPをパラメータに反映
            fxh2 = self._loss(X, T)  # 微小変化後の損失関数を計算
            # 偏微分の計算
            grad[idx] = (fxh1 - fxh2) / (2*h)
            # 微小変化させたPを元に戻す
            self.params[param_name][l] = P
        
        return grad.reshape(P.shape)

    def numerical_gradient_all(self, X, T):
        """
        ステップ2: 全パラメータの勾配を計算
        """
        grads = {'W': [],
                 'b': []}
        for l in range(self.n_layers):
            grads['W'].append(self._numerical_gradient(X, T, 'W', l))
            grads['b'].append(self._numerical_gradient(X, T, 'b', l))
        return grads

    def update_parameters(self, grads):
        """
        ステップ3: パラメータの更新
        """
        # パラメータの更新
        for l in range(self.n_layers):
            self.params['W'][l] -= learning_rate * grads['W'][l]
            self.params['b'][l] -= learning_rate * grads['b'][l]
    
    def fit(self, X, T, iters_num=1000):
        """
        ステップ4: ステップ1-3を繰り返す

        Parameters
        ----------
        X : numpy.ndarray 2D
            入力データ
        T : numpy.ndarray 1D or 2D
            正解データ
        iters_num : int
            学習 (SGD)の繰り返し数
        """
        # Tが1次元ベクトルなら2次元に変換してOne-hot encodingする
        T = self._one_hot_encoding(T)
        # iters_num繰り返す
        self.train_loss_list = []
        for i in range(iters_num):
            # ステップ1: ミニバッチの取得
            X_batch, T_batch = self.select_minibatch(X, T)
            # ステップ2: 勾配の計算
            grads = self.numerical_gradient_all(X_batch, T_batch)
            # ステップ3: パラメータの更新
            self.update_parameters(grads)
            # 学習経過の記録
            loss = network._loss(X_batch, T_batch)
            self.train_loss_list.append(loss)
        
    def accuracy(self, X_test, T_test):
        """
        正解率Accuracyを計算
        """
        # Tが1次元ベクトルなら2次元に変換してOne-hot encodingする
        T_test = self._one_hot_encoding(T_test)
        # 順伝播を計算
        Y_test = self._predict_onehot(X_test)
        Y_test_label = np.argmax(Y_test, axis=1)  # 予測クラス (One-hotをクラスのインデックスに変換)
        T_test_label = np.argmax(T_test, axis=1)  # 正解クラス (One-hotをクラスのインデックスに変換)
        accuracy = np.sum(Y_test_label == T_test_label) / float(X_test.shape[0])
        return accuracy

# データ読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'].isin(['versicolor', 'virginica'])]

# 説明変数
X = iris[['petal_width', 'petal_length', 'sepal_width']].to_numpy()
# 目的変数をOne-hot encoding
T = iris['species'].to_numpy()

# 学習データとテストデータ分割
X_train, X_test, T_train, T_test = train_test_split(X, T, shuffle=True, random_state=42)
train_size = X_train.shape[0]
# ハイパラ
iters_num = 4000  # 学習(SGD)の繰り返し数
hidden_size = 2  # 隠れ層のニューロン数
n_layers = 3  # 層数
batch_size = 50  # バッチサイズ (サンプリング数)
learning_rate = 1.0  # 学習率
weight_init_std=0.1

# ニューラルネットワーク計算用クラス
network = SGDNeuralNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                        loss_type='cross_entropy', weight_init_std=weight_init_std)
# SGDによる学習
network.fit(X_train, T_train, iters_num)
# 精度評価
print(f'{network.accuracy(X_test, T_test)}')
# 学習履歴のプロット
plt.plot(range(iters_num), network.train_loss_list)
plt.show()

# %% 決定境界のプロット
from seaborn_analyzer import classplot
import pandas as pd
network = SGDNeuralNet(X_train, T_train, hidden_size=hidden_size, n_layers=n_layers, 
                        loss_type='cross_entropy', weight_init_std=weight_init_std)
# 学習データをDataFrame化
iris_train = pd.DataFrame(np.column_stack([X_train, T_train]),
                columns=['petal_width', 'petal_length', 'sepal_width', 'species'])
# 決定境界をプロット
classplot.class_separator_plot(network, ['petal_width', 'petal_length', 'sepal_width'], 
                        'species', iris, fit_params={'iters_num': iters_num})


# %%
