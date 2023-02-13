import numpy as np

class Dropout:
    """
    Dropoutレイヤ
    """
    def __init__(self, dropout_ratio=0.5):
        """
        層の初期化

        Parameters
        ----------
        dropout_ratio : float
            Dropoutするニューロンの割合
        """
        self.dropout_ratio = dropout_ratio

    def _calc_output_shape(self):
        """層出力データの形状を計算"""
        # 層出力の形状は入力と等しい
        self.output_shape = self.input_shape

    def initialize_parameters(self, input_shape=None):
        """パラメータの初期化"""
        # input_shape引数が指定されているとき、メンバ変数に入力
        if input_shape is not None and self.input_shape is None:
            self.input_shape = input_shape
        # 層出力データの形状を計算
        self._calc_output_shape()
        # 消去ニューロン選択用maskの初期化
        self.mask = None

    def forward(self, Z_prev, train_flg=True):
        # 学習時 (消去したニューロンの位置を保存しておく)
        if train_flg:
            # ニューロンの消去位置を表すmask
            self.mask = np.random.rand(*Z_prev.shape) > self.dropout_ratio
            # 入力データにmaskを掛け合わせて該当ニューロンの出力をゼロに
            return Z_prev * self.mask
        # 推論時は、ニューロンの消去位置を保存しない
        else:
            return Z_prev * self.mask

    def backward(self, dZ):
        # 逆伝播入力dZにマスクを掛け、消去対象ニューロンの逆伝播出力をゼロに
        return dZ * self.mask