import numpy as np
from sklearn.preprocessing import OneHotEncoder
from common.loss_funcions import cross_entropy_error, squared_error
from common.forward_functions import forward_middle, forward_last_classification
from common.backward_functions import softmax_loss_backward, affine_backward_bias, affine_backward_weight, affine_backward_zprev, relu_backward, sigmoid_backward
from common.utils import calc_weight_init_std

class ConvolutionNet:
    def __init__(self, layers, 
                 batch_size, n_iter,
                 loss_type,
                 learning_rate, 
                 solver='sgd', momentum=0.9,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8):
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
        learning_rate : float
            学習率
        solver : {'sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'}
            最適化アルゴリズムの種類 ('sgd': SGD, 'momentum': モーメンタム, 'adagrad': AdaGrad, 'rmsprop': 'RMSProp', 'adam': Adam)
        momentum : float
            勾配移動平均の減衰率ハイパーパラメータ (solver = 'momentum'の時のみ有効)
        beta_1 : float
            勾配移動平均の減衰率ハイパーパラメータ (solver = 'adam'の時のみ有効)
        beta_2 : float
            過去の勾配2乗和の減衰率ハイパーパラメータ (solver = 'rmsprop' or 'adam'の時のみ有効)
        epsilon : float
            ゼロ除算によるエラーを防ぐハイパーパラメータ (solver = 'adagrad', 'rmsprop', or 'adam'の時のみ有効)
        """
        # 各種メンバ変数 (ハイパーパラメータ等)の入力
        self.layers = layers  # ネットワーク構造 (各層のクラスをリスト化したもの)
        self.learning_rate = learning_rate  # 学習率
        self.batch_size = batch_size  # ミニバッチのデータ数
        self.n_iter = n_iter  # 学習のイテレーション(繰り返し)数
        self.loss_type = loss_type  # 損失関数の種類
        self.solver = solver  # 最適化アルゴリズムの種類
        self.momentum = momentum  # 勾配移動平均の減衰率ハイパーパラメータ (モーメンタムで使用)
        self.beta_1 = beta_1  # 勾配移動平均の減衰率ハイパーパラメータ (Adamで使用)
        self.beta_2 = beta_2  # 過去の勾配2乗和の減衰率ハイパーパラメータ (RMSProp, Adamで使用)
        self.epsilon = epsilon  # ゼロ除算によるエラーを防ぐためのハイパーパラメータ (AdaGrad, RMSProp, Adamで使用)
        # 損失関数が正しく入力されているか判定
        if loss_type not in ['cross_entropy', 'squared_error']:
            raise Exception('the `loss_type` argument should be "cross_entropy" or "squared_error"')
        # パラメータを初期化
        self._initialize_parameters()
        # 層数を計算
        self.n_layers = len(self.layers)

    def _initialize_parameters(self):
        """
        パラメータを初期化
        """
        # 層ごとに初期化
        for l, layer in enumerate(self.layers):
            # 初層のとき、自身のinput_shapeを入力サイズとして使用
            if l == 0:
                layer.build(input_shape=layer.input_shape)
            # 初層以外のとき、前層の出力サイズを入力サイズとして使用
            else:
                layer.build(input_shape=self.layers[l-1].output_shape)
        # 最適化用の変数も初期化
        self._initialize_opt_params()

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
    
    def _predict_onehot(self, X, output_intermediate=False):
        """
        順伝播を全て計算(One-hot encodingで出力)
        """
        Z_current = X  # 入力値を保持
        # 順伝播
        for l, layer in enumerate(self.layers):
            Z_current = layer.forward(Z_current)
        #　結果を出力
        return Z_current
    
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
            return cross_entropy_error(Y, T)
        elif self.loss_type == 'squared_error':
            return squared_error(Y, T)
        else:
            raise Exception('The `loss_type` argument should be "cross_entropy" or "squared_error"')

    def gradient_backpropagation(self, X, T):
        """
        ステップ2: 誤差逆伝播法で全パラメータの勾配を計算
        """
        # 順伝播 (中間層出力Zおよび中間層の中間結果Aも保持する)
        Y = self._predict_onehot(X, output_intermediate=True)
        ###### 出力層の逆伝播 ######
        dZ = self.layers[self.n_layers-1].backward(Y, T)  # 逆伝播を計算
        ###### 中間層の逆伝播 (下流から順番にループ) ######
        for l in range(self.n_layers-2, -1, -1):
            dZ = self.layers[l].backward(dZ)  # 逆伝播を計算
    
    def _initialize_opt_params(self):
        """最適化で利用する変数の初期化"""
        # モーメンタムの勾配移動平均保持用変数momentum_v (最適化アルゴリズムがモーメンタム or Adamの時使用)
        if self.solver in ['momentum', 'adam']:
            self.momentum_v = [{} for layer in self.layers]  # 変数格納用の空の辞書のリスト
            for l, layer in enumerate(self.layers):  # 層ごとに初期化
                for k, param in layer.params.items():  # パラメータごとに初期化を実施
                    # self.paramsと同形状のndarrayのリストとして初期化 (全てゼロ埋め)
                    self.momentum_v[l][k] = np.zeros_like(param)
        # 過去の勾配2乗和保持用変数adagrad_h  (最適化アルゴリズムがAdaGrad, RMSProp, or Adamの時使用)
        if self.solver in ['adagrad', 'rmsprop', 'adam']:
            self.adagrad_h =[{} for layer in self.layers]
            for l, layer in enumerate(self.layers):  # 層ごとに初期化
                for k, param in layer.params.items():  # パラメータごとに初期化を実施
                    # self.paramsと同形状のndarrayのリストとして初期化 (全てゼロ埋め)
                    self.adagrad_h[l][k] = np.zeros_like(param)

    def _update_parameters_sgd(self):
        """SGDによるパラメータ更新"""
        # 層ごとにパラメータ更新
        for l, layer in enumerate(self.layers):
            # パラメータごとに更新
            for k, param in layer.params.items():
                # パラメータ更新量 = -学習率learning_rate * 勾配grads
                param -= self.learning_rate * layer.grad[k]

    def _update_parameters_momentum(self):
        """モーメンタムによるパラメータ更新"""
        # 層ごとにパラメータ更新
        for l, layer in enumerate(self.layers):
            # パラメータごとに更新
            for k, param in layer.params.items():
                # 勾配移動平均momentum_v = momentum * 更新前のmomentum_v - 学習率learning_rate * 勾配grads
                self.momentum_v[l][k] = self.momentum * self.momentum_v[l][k]- self.learning_rate * layer.grad[k]
                # パラメータ更新量 = momentum_v
                param += self.momentum_v[l][k]

    def _update_parameters_adagrad(self):
        """AdaGradによるパラメータ更新"""
        # 層ごとにパラメータ更新
        for l, layer in enumerate(self.layers):
            # パラメータごとに更新
            for k, param in layer.params.items():
                # 過去の勾配2乗和adagrad_h = 更新前のadagrad_h + 勾配gradsの2乗
                self.adagrad_h[l][k] = self.adagrad_h[l][k] + layer.grad[k] ** 2
                # パラメータ更新量 = -学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
                param -= self.learning_rate * layer.grad[k] / (np.sqrt(self.adagrad_h[l][k]) + self.epsilon)

    def _update_parameters_rmsprop(self):
        """RMSpropによるパラメータ更新"""
        # 層ごとにパラメータ更新
        for l, layer in enumerate(self.layers):
            # パラメータごとに更新
            for k, param in layer.params.items():
                # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
                self.adagrad_h[l][k] = self.beta_2 * self.adagrad_h[l][k] + (1 - self.beta_2) * layer.grad[k] ** 2
                # パラメータ更新量 = 学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
                param -= self.learning_rate * layer.grad[k] / (np.sqrt(self.adagrad_h[l][k]) + self.epsilon)

    def _update_parameters_adam(self):
        """Adamによるパラメータ更新"""
        # 層ごとにパラメータ更新
        for l, layer in enumerate(self.layers):
            # パラメータごとに更新
            for k, param in layer.params.items():
                # 勾配移動平均momentum_v = beta_1 * 更新前のmomentum_v - (1 - beta_1) * 勾配grads
                self.momentum_v[l][k] = self.beta_1 * self.momentum_v[l][k] + (1 - self.beta_1) * layer.grad[k]
                # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
                self.adagrad_h[l][k] = self.beta_2 * self.adagrad_h[l][k] + (1 - self.beta_2) * layer.grad[k] ** 2
                # パラメータ更新量 = 学習率learning_rate * momentum_v / (sqrt(adagrad_h)+epsilon)
                param -= self.learning_rate * self.momentum_v[l][k] / (np.sqrt(self.adagrad_h[l][k]) + self.epsilon)
    
    def update_parameters(self):
        """
        ステップ3: パラメータの更新
        """
        # 最適化アルゴリズムがSGDの時
        if self.solver == 'sgd':
            self._update_parameters_sgd()
        # 最適化アルゴリズムがモーメンタムの時
        elif self.solver == 'momentum':
            self._update_parameters_momentum()
        # 最適化アルゴリズムがAdaGradの時
        elif self.solver == 'adagrad':
            self._update_parameters_adagrad()
        # 最適化アルゴリズムがRMSpropの時
        elif self.solver == 'rmsprop':
            self._update_parameters_rmsprop()
        # 最適化アルゴリズムがAdamの時
        elif self.solver == 'adam':
            self._update_parameters_adam()

    def fit(self, X, T):
        """
        ステップ4: ステップ1-3を繰り返す

        Parameters
        ----------
        X : numpy.ndarray 2D
            入力データ
        T : numpy.ndarray 1D or 2D
            正解データ
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
            self.update_parameters()
            # 学習経過の記録
            loss = self._loss(X_batch, T_batch)
            self.train_loss_list.append(loss)
            # 学習経過をプロット
            if i_iter%10 == 0:
                print(f'Iteration{i_iter}/{self.n_iter}')
        
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