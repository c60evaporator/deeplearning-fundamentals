import numpy as np
from common.forward_functions import forward_middle, forward_last_classification
from common.backward_functions import softmax_loss_backward, affine_backward_bias, affine_backward_weight, affine_backward_zprev, relu_backward, sigmoid_backward
from common.utils import calc_weight_init_std

class Dense():
    """
    中間層の全結合層 (Affineレイヤ+活性化関数レイヤ)
    """
    def __init__(self, units=32, activation_function='relu', weight_init_std='auto', input_shape=None):
        """
        層の初期化

        Parameters
        ----------
        units : int
            ニューロン数
        activation_function : {'sigmoid', 'relu'}
            中間層活性化関数の種類 ('sigmoid': シグモイド関数, 'relu': ReLU関数)
        weight_init_std : float or 'auto'
            重み初期値生成時の標準偏差 ('auto'を指定すると、activation_function='sigmoid'の時Xavierの初期値を、'relu'の時Heの初期値を使用)
        input_shape : tuple
            入力データの形状 (初層のみ入力が必要, データ数は形状に含めない)
        """
        self.units = units
        self.activation_function = activation_function
        self.weight_init_std = weight_init_std
        self.input_shape = input_shape

    def _calc_output_shape(self):
        """層出力データの形状を計算"""
        # 層出力の形状はunitsと等しい (=ニューロン数)
        self.output_shape = (self.units,)

    def initialize_parameters(self, input_shape=None):
        """パラメータの初期化"""
        # input_shape引数が指定されているとき、メンバ変数に入力
        if input_shape is not None and self.input_shape is None:
            self.input_shape = input_shape
        # 層出力データの形状を計算
        self._calc_output_shape()
        # パラメータ初期化(重み+バイアス)
        self.params={}
        self.params['W'] = calc_weight_init_std(self.input_shape[-1], self.weight_init_std, self.activation_function) * \
                            np.random.randn(self.input_shape[-1], self.units)
        self.params['b'] = np.zeros(self.units)

    def forward(self, Z_prev, train_flg=None):
        """順伝播"""
        self.Z_prev = Z_prev  # 入力を保持 (逆伝播で使用)
        # 順伝播を計算(Affineレイヤ出力A、活性化関数レイヤ出力Zはメンバ変数に保持)
        self.Z, self.A = forward_middle(Z_prev, self.params['W'], self.params['b'], 
                activation_function=self.activation_function, output_A=True)
        return self.Z

    def backward(self, dZ):
        """逆伝播"""
        # 勾配保持用
        self.grads={}
        # Reluレイヤ
        if self.activation_function == 'relu':
            dA = relu_backward(dZ, self.A)  # (Affineレイヤ出力Aを入力)
        # Sigmoidレイヤ
        if self.activation_function == 'sigmoid':
            dA = sigmoid_backward(dZ, self.Z)  # (活性化関数レイヤ出力Zを入力)
        # Affineレイヤ
        db = affine_backward_bias(dA)  # バイアスパラメータbの偏微分
        dW = affine_backward_weight(dA, self.Z_prev)  # 重みパラメータZの偏微分 (前層出力Z_prevを入力)
        dZ_prev = affine_backward_zprev(dA, self.params['W'])  # 前層出力dZ_prevの偏微分 (重みパラメータWを入力)
        # 計算した偏微分(勾配)を保持
        self.grads['b'] = db
        self.grads['W'] = dW
        # 前層出力の偏微分(勾配)dZ_prevを出力
        return dZ_prev

class DenseOutput(Dense):
    """
    出力層の全結合層 (Affineレイヤ+活性化関数レイヤ)
    """
    def __init__(self, units=2, activation_function='softmax', weight_init_std='auto'):
        """
        層の初期化

        Parameters
        ----------
        units : int
            ニューロン数
        activation_function : {'softmax', 'identity'}
            出力層活性化関数の種類 ('softmax': ソフトマックス関数, 'identity': 恒等関数)
        weight_init_std : float or 'auto'
            重み初期値生成時の標準偏差 ('auto'を指定すると、activation_function='sigmoid'の時Xavierの初期値を、'relu'の時Heの初期値を使用)
        """
        super().__init__(units=units, activation_function=activation_function, 
                         weight_init_std=weight_init_std)

    def forward(self, Z_prev, train_flg=None):
        """順伝播"""
        self.Z_prev = Z_prev  # 入力を保持 (逆伝播で使用)
        # 順伝播を計算(Affineレイヤ出力A、活性化関数レイヤ出力Zはメンバ変数に保持)
        self.Z = forward_last_classification(Z_prev, self.params['W'], self.params['b'])
        return self.Z

    def backward(self, Y, T):
        """逆伝播"""
        # 勾配保持用
        self.grads={}
        # Softmax-with-Lossレイヤ
        dA = softmax_loss_backward(Y, T)
        # Affineレイヤ
        db = affine_backward_bias(dA)  # バイアスパラメータbの偏微分
        dW = affine_backward_weight(dA, self.Z_prev)  # 重みパラメータZの偏微分 (前層出力Z_prevを入力)
        dZ_prev = affine_backward_zprev(dA, self.params['W'])  # 前層出力dZ_prevの偏微分 (重みパラメータWを入力)
        # 計算した偏微分(勾配)を保持
        self.grads['b'] = db
        self.grads['W'] = dW
        # 前層出力の偏微分(勾配)dZ_prevを出力
        return dZ_prev