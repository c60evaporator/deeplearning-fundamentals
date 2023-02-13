import numpy as np
from common.forward_advanced import forward_batch_normalization
from common.backward_advanced import backward_batch_normalization
from common.utils import flatten, calc_flatten_shape

class BatchNormalization:
    """
    Batch Normalizationレイヤ
    """
    def __init__(self, momentum=0.99, epsilon=1e-6, input_shape=None):
        """
        層の初期化

        Parameters
        ----------
        momentum : float
            muとvarの移動平均減衰率ハイパーパラメータ
        epsilon : float
            標準化でゼロ除算を防ぐための微小値
        input_shape : tuple
            入力データの形状 (初層のみ入力が必要, データ数は形状に含めない)
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_shape = input_shape

    def _calc_output_shape(self):
        """層出力データの形状を計算"""
        # 層出力の形状は入力と等しい
        self.output_shape = self.input_shape
        # 画像変換後のデータ形状も計算 (データ数を除いた2次元以上→1次元)
        self.reshaped_shape = calc_flatten_shape(self.input_shape)

    def initialize_parameters(self, input_shape=None):
        """パラメータの初期化"""
        # input_shape引数が指定されているとき、メンバ変数に入力
        if input_shape is not None and self.input_shape is None:
            self.input_shape = input_shape
        # 層出力データの形状を計算
        self._calc_output_shape()
        # パラメータ初期化(スケールgamma+シフトbeta)
        self.params={}
        self.params['gamma'] = np.ones(self.reshaped_shape)
        self.params['beta'] = np.zeros(self.reshaped_shape)
        # muとvarの移動平均も初期化
        self.running_mean = np.zeros(self.reshaped_shape)
        self.running_var = np.zeros(self.reshaped_shape)

    def forward(self, Z_prev, train_flg=True):
        # 画像(2次元以上)を1次元配列に変換 (バッチデータも含めて2次元に変換)
        Z_prev_flatten = flatten(Z_prev)
        # 学習時 (muとvarに生値を使用＆移動平均を計算)
        if train_flg:
            # 順伝播を実行
            self.Z, self.A, self.var, self.Zc, self.mu = forward_batch_normalization(
                Z_prev_flatten, self.params['gamma'], self.params['beta'], self.epsilon)
            # muとvarの移動平均を計算
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * self.var
            # 入力データの形状に再変換して返す
            return self.Z.reshape(*Z_prev.shape)

        # 推論時は、muとvarに学習で求めた移動平均を使用
        else:
            Zc = Z_prev - self.running_mean
            A = Zc / np.sqrt(self.running_var + self.epsilon)
            Z = self.gamma * A + self.beta
            # 入力データの形状に再変換して返す
            return Z.reshape(*Z_prev.shape)

    def backward(self, dZ):
        # 画像データの変換 (データ数を除いて2次元以上→1次元のベクトルに変換)
        dZ_flatten = flatten(dZ)
        # 逆伝播を計算
        dZ_prev, dgamma, dbeta = backward_batch_normalization(dZ_flatten, self.gamma, self.batch_size, self.A, self.var, self.Zc)
        # パラメータの偏微分を保持
        self.dgamma = dgamma
        self.dbeta = dbeta
        # 入力データの形状に再変換して返す
        return dZ_prev.reshape(*dZ.shape)