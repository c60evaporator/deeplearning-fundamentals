import numpy as np
from sklearn.preprocessing import OneHotEncoder
from common.loss_funcions import cross_entropy_error, squared_error
from common.forward_functions import forward_middle, forward_last_classification
from common.backward_functions import softmax_loss_backward, affine_backward_bias, affine_backward_weight, affine_backward_zprev, relu_backward, sigmoid_backward
from common.utils import calc_weight_init_std

class BackpropAdvancedNet:
    def __init__(self, X, T, 
                 hidden_size, n_layers,
                 batch_size, n_iter,
                 loss_type, activation_function,
                 learning_rate, solver='sgd', momentum=0.9,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 weight_decay_lambda=0,
                 weight_init_std='auto'):
        """
        ハイパーパラメータの読込＆パラメータの初期化

        Parameters
        ----------
        X : numpy.ndarray 2D
            入力データ (データの次元数確認のみに使用)
        T : numpy.ndarray 1D or 2D
            正解データ (データの次元数確認のみに使用)
        hidden_size : int
            隠れ層の1層あたりニューロン
        n_layers : int
            層数 (隠れ層の数 - 1)
        batch_size : int
            ミニバッチのデータ数
        n_iter : int
            学習 (SGD)の繰り返し数
        loss_type : {'cross_entropy', 'squared_error'}
            損失関数の種類 ('cross_entropy': 交差エントロピー誤差, 'squared_error': 2乗和誤差)
        activation_function : {'sigmoid', 'relu'}
            中間層活性化関数の種類 ('sigmoid': シグモイド関数, 'relu': ReLU関数)
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
        weight_decay_lambda : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        weight_init_std : float or 'auto'
            重み初期値生成時の標準偏差 ('auto'を指定すると、activation_function='sigmoid'の時Xavierの初期値を、'relu'の時Heの初期値を使用)
        """
        # 各種メンバ変数 (ハイパーパラメータ等)の入力
        self.input_size = X.shape[1]  # 説明変数の次元数(1層目の入力数)
        self.output_size = T.shape[1] if T.ndim == 2 else np.unique(T).size  # クラス数 (出力層のニューロン数)
        self.hidden_size = hidden_size  # 隠れ層の1層あたりニューロン
        self.n_layers = n_layers  # 層数
        self.batch_size = batch_size  # ミニバッチのデータ数
        self.n_iter = n_iter  # 学習のイテレーション(繰り返し)数
        self.loss_type = loss_type  # 損失関数の種類
        self.activation_function = activation_function  # 中間層活性化関数の種類
        self.learning_rate = learning_rate  # 学習率
        self.solver = solver  # 最適化アルゴリズムの種類
        self.momentum = momentum  # 勾配移動平均の減衰率ハイパーパラメータ (モーメンタムで使用)
        self.beta_1 = beta_1  # 勾配移動平均の減衰率ハイパーパラメータ (Adamで使用)
        self.beta_2 = beta_2  # 過去の勾配2乗和の減衰率ハイパーパラメータ (RMSProp, Adamで使用)
        self.epsilon = epsilon  # ゼロ除算によるエラーを防ぐためのハイパーパラメータ (AdaGrad, RMSProp, Adamで使用)
        self.weight_decay_lambda = weight_decay_lambda  # Weight decayの正則化効果の強さを表すハイパーパラメータ
        self.weight_init_std = weight_init_std  # 重み初期値生成時の標準偏差
        # 損失関数と活性化関数が正しく入力されているか判定
        if loss_type not in ['cross_entropy', 'squared_error']:
            raise Exception('the `loss_type` argument should be "cross_entropy" or "squared_error"')
        if activation_function not in ['sigmoid', 'relu']:
            raise Exception('the `activation_function` argument should be "sigmoid" or "relu"')
        # パラメータを初期化
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        パラメータを初期化
        """
        # パラメータ格納用に空の辞書のリストを準備
        self.params = [{} for l in range(self.n_layers)]
        # 重みパラメータ
        self.params[0]['W'] = calc_weight_init_std(self.input_size, self.weight_init_std, self.activation_function) \
                            * np.random.randn(self.input_size, self.hidden_size)  # 1層目の重みパラメータ
        for l in range(1, self.n_layers-1):
            self.params[l]['W'] = calc_weight_init_std(self.hidden_size, self.weight_init_std, self.activation_function) \
                            * np.random.randn(self.hidden_size, self.hidden_size) # 中間層の重みパラメータ
        self.params[self.n_layers-1]['W'] = calc_weight_init_std(self.hidden_size, self.weight_init_std, 'softmax') \
                            * np.random.randn(self.hidden_size, self.output_size) # 出力層の重みパラメータ
        # バイアスパラメータ
        for l in range(self.n_layers-1):
            self.params[l]['b'] = np.zeros(self.hidden_size)  # 中間層のバイアスパラメータ
        self.params[self.n_layers-1]['b'] = np.zeros(self.output_size)  # 最終層のバイアスパラメータ
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
    
    def _predict_onehot(self, X, train_flg=False):
        """
        順伝播を全て計算(One-hot encodingで出力)
        """
        Z_current = X  # 入力値を保持
        Z_intermediate = []  # 中間層出力の保持用 (5章の誤差逆伝播法で使用)
        A_intermediate = []  # 中間層の途中結果Aの保持用 (5章の誤差逆伝播法で使用)
        # 中間層(1〜n_layers-1層目)の順伝播
        for l in range(self.n_layers-1):
            W = self.params[l]['W']  # 重みパラメータ
            b = self.params[l]['b']  # バイアスパラメータ
            Z_current, A_current = forward_middle(Z_current, W, b, 
                activation_function=self.activation_function, output_A=True)  # 中間層の計算
            Z_intermediate.append(Z_current)  # 中間層出力を保持 (5章の誤差逆伝播法で使用)
            A_intermediate.append(A_current)  # 中間層の途中結果Aを保持 (5章の誤差逆伝播法で使用)
        # 出力層の順伝播
        W_final = self.params[self.n_layers-1]['W']
        b_final = self.params[self.n_layers-1]['b']
        Z_result = forward_last_classification(Z_current, W_final, b_final)
        # 中間層出力も出力する場合 (5章の誤差逆伝播法で使用)
        if train_flg:
            return Z_result, Z_intermediate, A_intermediate
        # 中間層出力を出力しない場合
        else:
            return Z_result
    
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
            loss = cross_entropy_error(Y, T)
        elif self.loss_type == 'squared_error':
            loss = squared_error(Y, T)
        else:
            raise Exception('The `loss_type` argument should be "cross_entropy" or "squared_error"')
        # Weight decayの計算
        weight_decay = 0
        for l in range(self.n_layers):
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params[l]['W'] ** 2)
        # 元の損失関数 + Weight decayを返す
        return loss + weight_decay

    def gradient_backpropagation(self, X, T):
        """
        ステップ2: 誤差逆伝播法で全パラメータの勾配を計算
        """
        # 順伝播 (中間層出力Zおよび中間層の中間結果Aも保持する)
        Y, Z_intermediate, A_intermediate = self._predict_onehot(X, train_flg=True)
        # 逆伝播結果格納用 (空の辞書のリスト)
        grads = [{} for l in range(self.n_layers)]
        ###### 出力層の逆伝播 ######
        # Softmax-with-Lossレイヤ
        dA = softmax_loss_backward(Y, T)
        # Affineレイヤ
        db = affine_backward_bias(dA)  # バイアスパラメータbの偏微分
        dW = affine_backward_weight(dA, Z_intermediate[self.n_layers-2])  # 重みパラメータWの偏微分 (前層出力Z_prevを入力)
        dZ_prev = affine_backward_zprev(dA, self.params[self.n_layers-1]['W'])  # 前層出力Z_prevの偏微分 (重みパラメータWを入力)
        # 計算した偏微分(勾配)を保持
        grads[self.n_layers-1]['b'] = db
        grads[self.n_layers-1]['W'] = dW \
            + self.weight_decay_lambda * self.params[self.n_layers-1]['W']  # Weight decay分を勾配に足す
        ###### 中間層の逆伝播 (下流から順番にループ) ######
        for l in range(self.n_layers-2, -1, -1):
            # 当該層の出力偏微分dZを更新
            dZ = dZ_prev.copy()
            # Reluレイヤ
            if self.activation_function == 'relu':
                dA = relu_backward(dZ, A_intermediate[l])  # (中間結果Aを入力)
            # Sigmoidレイヤ
            if self.activation_function == 'sigmoid':
                dA = sigmoid_backward(dZ, Z_intermediate[l])  # (中間層出力Zを入力)
            # Affineレイヤ
            db = affine_backward_bias(dA)  # バイアスパラメータbの偏微分
            # 初層以外の場合
            if l > 0:
                dW = affine_backward_weight(dA, Z_intermediate[l-1])  # 重みパラメータWの偏微分 (前層出力Z_prevを入力)
                dZ_prev = affine_backward_zprev(dA, self.params[l]['W'])  # 前層出力Z_prevの偏微分 (重みパラメータWを入力)
            # 初層の場合
            else:
                dW = affine_backward_weight(dA, X)  # 重みパラメータZ (入力データXを入力)
            # 計算した偏微分(勾配)を保持
            grads[l]['b'] = db
            grads[l]['W'] = dW \
                + self.weight_decay_lambda * self.params[l]['W']  # Weight decay分を勾配に足す

        return grads
    
    def _initialize_opt_params(self):
        """最適化で利用する変数の初期化"""
        # モーメンタムの勾配移動平均保持用変数momentum_v (最適化アルゴリズムがモーメンタム or Adamの時使用)
        if self.solver in ['momentum', 'adam']:
            self.momentum_v = [{} for l in range(self.n_layers)]  # 変数格納用の空の辞書のリスト
            for l in range(self.n_layers):  # 層ごとに初期化
                # self.paramsと同形状のndarrayのリストとして初期化 (全てゼロ埋め)
                self.momentum_v[l]['W'] = np.zeros_like(self.params[l]['W'])
                self.momentum_v[l]['b'] = np.zeros_like(self.params[l]['b'])
        # 過去の勾配2乗和保持用変数adagrad_h  (最適化アルゴリズムがAdaGrad, RMSProp, or Adamの時使用)
        if self.solver in ['adagrad', 'rmsprop', 'adam']:
            self.adagrad_h =[{} for l in range(self.n_layers)]  # 変数格納用の空の辞書のリスト
            for l in range(self.n_layers):  # 層ごとに初期化
                # self.paramsと同形状のndarrayのリストとして初期化 (全てゼロ埋め)
                self.adagrad_h[l]['W'] = np.zeros_like(self.params[l]['W'])
                self.adagrad_h[l]['b'] = np.zeros_like(self.params[l]['b'])

    def _update_parameters_sgd(self, grads):
        """SGDによるパラメータ更新"""
        for l in range(self.n_layers):
            # パラメータ更新量 = -学習率learning_rate * 勾配grads
            self.params[l]['W'] -= self.learning_rate * grads[l]['W']
            self.params[l]['b'] -= self.learning_rate * grads[l]['b']

    def _update_parameters_momentum(self, grads):
        """モーメンタムによるパラメータ更新"""
        for l in range(self.n_layers):
            # 勾配移動平均momentum_v = momentum * 更新前のmomentum_v - 学習率learning_rate * 勾配grads
            self.momentum_v[l]['W'] = self.momentum * self.momentum_v[l]['W'] - self.learning_rate * grads[l]['W']
            self.momentum_v[l]['b'] = self.momentum * self.momentum_v[l]['b'] - self.learning_rate * grads[l]['b']
            # パラメータ更新量 = momentum_v
            self.params[l]['W'] += self.momentum_v[l]['W']
            self.params[l]['b'] += self.momentum_v[l]['b']

    def _update_parameters_adagrad(self, grads):
        """AdaGradによるパラメータ更新"""
        for l in range(self.n_layers):
            # 過去の勾配2乗和adagrad_h = 更新前のadagrad_h + 勾配gradsの2乗
            self.adagrad_h[l]['W'] = self.adagrad_h[l]['W'] + grads[l]['W'] ** 2
            self.adagrad_h[l]['b'] = self.adagrad_h[l]['b'] + grads[l]['b'] ** 2
            # パラメータ更新量 = -学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
            self.params[l]['W'] -= self.learning_rate * grads[l]['W'] / (np.sqrt(self.adagrad_h[l]['W']) + self.epsilon)
            self.params[l]['b'] -= self.learning_rate * grads[l]['b'] / (np.sqrt(self.adagrad_h[l]['b']) + self.epsilon)

    def _update_parameters_rmsprop(self, grads):
        """RMSpropによるパラメータ更新"""
        for l in range(self.n_layers):
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[l]['W'] = self.beta_2 * self.adagrad_h[l]['W'] + (1 - self.beta_2) * grads[l]['W'] ** 2
            self.adagrad_h[l]['b'] = self.beta_2 * self.adagrad_h[l]['b'] + (1 - self.beta_2) * grads[l]['b'] ** 2
            # パラメータ更新量 = 学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
            self.params[l]['W'] -= self.learning_rate * grads[l]['W'] / (np.sqrt(self.adagrad_h[l]['W']) + self.epsilon)
            self.params[l]['b'] -= self.learning_rate * grads[l]['b'] / (np.sqrt(self.adagrad_h[l]['b']) + self.epsilon)

    def _update_parameters_adam(self, grads):
        """Adamによるパラメータ更新"""
        for l in range(self.n_layers):
            # 勾配移動平均momentum_v = beta_1 * 更新前のmomentum_v - (1 - beta_1) * 勾配grads
            self.momentum_v[l]['W'] = self.beta_1 * self.momentum_v[l]['W'] + (1 - self.beta_1) * grads[l]['W']
            self.momentum_v[l]['b'] = self.beta_1 * self.momentum_v[l]['b'] + (1 - self.beta_1) * grads[l]['b']
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[l]['W'] = self.beta_2 * self.adagrad_h[l]['W'] + (1 - self.beta_2) * grads[l]['W'] ** 2
            self.adagrad_h[l]['b'] = self.beta_2 * self.adagrad_h[l]['b'] + (1 - self.beta_2) * grads[l]['b'] ** 2
            # パラメータ更新量 = 学習率learning_rate * momentum_v / (sqrt(adagrad_h)+epsilon)
            self.params[l]['W'] -= self.learning_rate * self.momentum_v[l]['W'] / (np.sqrt(self.adagrad_h[l]['W']) + self.epsilon)
            self.params[l]['b'] -= self.learning_rate * self.momentum_v[l]['b'] / (np.sqrt(self.adagrad_h[l]['b']) + self.epsilon)
    
    def update_parameters(self, grads):
        """
        ステップ3: パラメータの更新
        """
        # 最適化アルゴリズムがSGDの時
        if self.solver == 'sgd':
            self._update_parameters_sgd(grads)
        # 最適化アルゴリズムがモーメンタムの時
        elif self.solver == 'momentum':
            self._update_parameters_momentum(grads)
        # 最適化アルゴリズムがAdaGradの時
        elif self.solver == 'adagrad':
            self._update_parameters_adagrad(grads)
        # 最適化アルゴリズムがRMSpropの時
        elif self.solver == 'rmsprop':
            self._update_parameters_rmsprop(grads)
        # 最適化アルゴリズムがAdamの時
        elif self.solver == 'adam':
            self._update_parameters_adam(grads)

    
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
            grads = self.gradient_backpropagation(X_batch, T_batch)
            # ステップ3: パラメータの更新
            self.update_parameters(grads)
            # 学習経過の記録
            loss = self._loss(X_batch, T_batch)
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