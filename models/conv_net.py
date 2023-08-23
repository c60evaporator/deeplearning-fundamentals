import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder
import copy

from common.loss_funcions import cross_entropy_error, squared_error
from common.optimizers import BaseOptimizer, SGD, Momentum, AdaGrad, RMSprop, Adam, AdamW

class ConvolutionNet:
    def __init__(self, layers: List, 
                 batch_size: int, n_iter: int,
                 loss_type: str,
                 optimizer='adam'):
        """
        ハイパーパラメータの読込＆パラメータの初期化

        Parameters
        ----------
        layers : list
            ネットワーク構造 (各層のクラスをリスト化したもの)
        batch_size : int
            ミニバッチのデータ数
        n_iter : int
            学習 (SGD)の繰り返し数
        loss_type : {'cross_entropy', 'squared_error'}
            損失関数の種類 ('cross_entropy': 交差エントロピー誤差, 'squared_error': 2乗和誤差)
        optimizer : {common.optimizers.BaseOptimizer, 'sgd', 'momentum', 'adagrad', 'rmsprop', 'adam', 'adamw'}
            最適化アルゴリズムの種類 ('sgd': SGD, 'momentum': モーメンタム, 'adagrad': AdaGrad, 'rmsprop': 'RMSProp', 'adam': Adam, 'adamw': AdamW)
        """
        # 各種メンバ変数 (ハイパーパラメータ等)の入力
        self.layers = layers  # ネットワーク構造 (各層のクラスをリスト化したもの)
        self.batch_size = batch_size  # ミニバッチのデータ数
        self.n_iter = n_iter  # 学習のイテレーション(繰り返し)数
        self.loss_type = loss_type  # 損失関数の種類
        self.optimizer = optimizer  # 最適化アルゴリズムの種類

        # 損失関数が正しく入力されているか判定
        if loss_type not in ['cross_entropy', 'squared_error']:
            raise Exception('the `loss_type` argument should be "cross_entropy" or "squared_error"')
        # パラメータを初期化
        self._initialize_parameters()
        # 層数を計算
        self.n_layers = len(self.layers)

    def _initialize_parameters(self):
        """
        パラメータ等を初期化
        """
        # 層ごとにパラメータ初期化
        for l, layer in enumerate(self.layers):
            # 初層のとき、クラス初期化時に指定したinput_shapeを入力サイズとして使用
            if l == 0:
                layer.initialize_parameters()
            # 初層以外のとき、前層の出力サイズを入力サイズとして使用
            else:
                layer.initialize_parameters(input_shape=self.layers[l-1].output_shape)

        # 最適化用クラスの初期化
        self._initialize_optimizers()

        # 全結合層のとき、層ごとにWeight decay係数を適用
        for l, layer in enumerate(self.layers):
            if 'W' in layer.params:
                # 最適化アルゴリズムがAdamWのとき、損失関数にWeight decayは適用しない
                if isinstance(self.optimizers[l], AdamW):
                    layer.weight_decay = 0
                # AdamW以外のとき、OptimizerのWeight decayの値を保持
                layer.weight_decay = self.optimizers[l].weight_decay

    def _initialize_optimizers(self):
        """最適化で利用するクラスの初期化"""
        self.optimizers=[]  # 相互ごとの最適化用インスタンス保持用のリスト
        for l, layer in enumerate(self.layers):  # 層ごとに初期化
            # 最適化をクラスで指定したとき
            if isinstance(self.optimizer, BaseOptimizer):
                self.optimizers.append(copy.deepcopy(self.optimizer))
            # 最適化を文字列で指定したとき (SGD)
            if self.optimizer == 'sgd':
                self.optimizers.append(SGD())
            # 最適化を文字列で指定したとき (モーメンタム)
            elif self.optimizer == 'momentum':
                self.optimizers.append(Momentum())
            # 最適化を文字列で指定したとき (AdaGrad)
            elif self.optimizer == 'adagrad':
                self.optimizers.append(AdaGrad())
            # 最適化を文字列で指定したとき (RMSprop)
            elif self.optimizer == 'rmsprop':
                self.optimizers.append(RMSprop())
            # 最適化を文字列で指定したとき (Adam)
            elif self.optimizer == 'adam':
                self.optimizers.append(Adam())
            # 最適化を文字列で指定したとき (AdamW)
            elif self.optimizer == 'adamw':
                self.optimizers.append(AdamW())
            
            # 最適化で使用する変数の初期化
            self.optimizers[l].initialize_opt_params(layer.params)

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
    
    def _predict_onehot(self, X, train_flg=False):
        """
        順伝播を全て計算(One-hot encodingで出力)
        """
        Z_current = X  # 入力値を保持
        # 順伝播
        for l, layer in enumerate(self.layers):
            Z_current = layer.forward(Z_current, train_flg)
        #　結果を出力
        return Z_current
    
    def predict(self, X):
        """
        順伝播を全て計算(クラス名で出力)

        Parameters
        ----------
        X : np.ndarray
            入力データとなる2D numpy配列（形状:(n_samples, n_features)）
        
        Returns
        -------
        np.ndarray
            予測されたクラスラベルの1D numpy配列
        """
        Y = self._predict_onehot(X)
        Y = self._one_hot_encoding_reverse(Y)
        return Y

    def select_minibatch(self, X: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ステップ1: ミニバッチの取得

        Parameters
        ----------
        X : np.ndarray
            入力データとなる2D numpy配列（形状:(n_samples, n_features)）
        T : np.ndarray
            ターゲットラベルとなる2D numpy配列（形状:(n_samples, n_classes)）

        Returns
        -------
        X_batch : np.ndarray
            ランダムに選択されたミニバッチの入力データ（形状：(batch_size, n_features)）
        T_batch : np.ndarray
            ランダムに選択されたミニバッチの教師データ（形状：(batch_size, n_classes)）
        """
        train_size = X.shape[0]  # サンプリング前のデータ数
        batch_mask = np.random.choice(train_size, self.batch_size)  # ランダムサンプリング
        X_batch = X[batch_mask]
        T_batch = T[batch_mask]
        return X_batch, T_batch

    def _loss(self, X, T):
        """
        損失関数の計算
        """
        Y = self._predict_onehot(X)
        if self.loss_type == 'cross_entropy':
            loss = cross_entropy_error(Y, T)
        elif self.loss_type == 'squared_error':
            loss = squared_error(Y, T)
        else:
            raise Exception('The `loss_type` argument should be "cross_entropy" or "squared_error"')
        # Weight decayの計算
        weight_decay = 0
        for l, layer in enumerate(self.layers):
            if 'W' in layer.params:
                weight_decay += 0.5 * layer.weight_decay * np.sum(layer.params['W'] ** 2)
        # 元の損失関数 + Weight decayを返す
        return loss + weight_decay

    def gradient_backpropagation(self, X: np.ndarray, T: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        ステップ2: 誤差逆伝播法で全パラメータの勾配を計算

        Parameters
        ----------
        X : np.ndarray
            入力データとなる2D numpy配列（形状:(n_samples, n_features)）
        T : np.ndarray
            ターゲットラベルとなる2D numpy配列（形状:(n_samples, n_classes)）

        Returns
        -------
        grads : List[Dict[str, np.ndarray]]
            計算された各層の勾配をリストとして保持 (パラメータ名をキーとした辞書のリスト)
        """
        # 順伝播 (中間層出力Zおよび中間層の中間結果Aも保持する)
        Y = self._predict_onehot(X, train_flg=True)
        ###### 出力層の逆伝播 ######
        dZ = self.layers[self.n_layers-1].backward(Y, T)  # 逆伝播を計算
        ###### 中間層の逆伝播 (下流から順番にループ) ######
        for l in range(self.n_layers-2, -1, -1):
            dZ = self.layers[l].backward(dZ)  # 逆伝播を計算

    def update_parameters(self, i_iter: int):
        """
        ステップ3: パラメータの更新

        Parameters
        ----------
        i_iter : int
            現在の学習イテレーション(繰り返し)数
        """
        # 層ごとに最適化アルゴリズムによるパラメータ更新を実施
        for l, layer in enumerate(self.layers):
            self.optimizers[l].update(layer.params, layer.grads, i_iter)

    def fit(self, X: np.ndarray, T: np.ndarray):
        """
        ステップ4: ステップ1-3を繰り返す

        Parameters
        ----------
        X : np.ndarray
            入力データとなる2D numpy配列（形状:(n_samples, n_features)）
        T : np.ndarray
            ターゲットラベルとなる1D or 2D numpy配列（1次元ベクトルの場合One-hot encodingで自動変換される）
        """
        # パラメータを初期化
        self._initialize_parameters()
        # Tが1次元ベクトルなら2次元に変換してOne-hot encodingする
        T = self._one_hot_encoding(T)
        # n_iter繰り返す
        self.train_loss_list = []
        for i_iter in range(self.n_iter):
            # ステップ1: ミニバッチの取得
            X_batch, T_batch = self.select_minibatch(X, T)
            # ステップ2: 勾配の計算
            self.gradient_backpropagation(X_batch, T_batch)
            # ステップ3: パラメータの更新
            self.update_parameters(i_iter)
            # 学習経過の記録
            loss = self._loss(X_batch, T_batch)
            self.train_loss_list.append(loss)
            # 学習経過をプロット
            if i_iter%10 == 0:
                print(f'Iteration{i_iter}/{self.n_iter}')
        
    def accuracy(self, X_test, T_test) -> float:
        """
        正解率Accuracyを計算

        Parameters
        ----------
        X_test : np.ndarray
            入力データとなる2D numpy配列（形状:(n_samples, n_features)）
        T_test : np.ndarray
            ターゲットラベルとなる1D or 2D numpy配列（1次元ベクトルの場合One-hot encodingで自動変換される）

        Returns
        -------
        float
            正解率 (Accuracy)
        """
        # Tが1次元ベクトルなら2次元に変換してOne-hot encodingする
        T_test = self._one_hot_encoding(T_test)
        # 順伝播を計算
        Y_test = self._predict_onehot(X_test)
        Y_test_label = np.argmax(Y_test, axis=1)  # 予測クラス (One-hotをクラスのインデックスに変換)
        T_test_label = np.argmax(T_test, axis=1)  # 正解クラス (One-hotをクラスのインデックスに変換)
        accuracy = np.sum(Y_test_label == T_test_label) / float(X_test.shape[0])
        return accuracy