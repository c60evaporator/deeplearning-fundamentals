import numpy as np

class SGD:
    """SGDによる最適化クラス"""
    def __init__(self, learning_rate):
        """クラスの初期化"""
        self.learning_rate = learning_rate
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        pass
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            params[k] -= self.learning_rate * grads[k]

class Momentum:
    """モーメンタムによる最適化クラス"""
    def __init__(self, learning_rate, momentum):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.momentum_v = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均momentum_v = momentum * 更新前のmomentum_v - 学習率learning_rate * 勾配grads
            self.momentum_v[k] = self.momentum * self.momentum_v[k]- self.learning_rate * grads[k]
            # パラメータ更新量 = momentum_v
            params[k] += self.momentum_v[k]

class AdaGrad:
    """AdaGradによる最適化クラス"""
    def __init__(self, learning_rate, epsilon):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.adagrad_h = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 過去の勾配2乗和adagrad_h = 更新前のadagrad_h + 勾配gradsの2乗
            self.adagrad_h[k] = self.adagrad_h[k] + grads[k] ** 2
            # パラメータ更新量 = -学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
            params[k] -= self.learning_rate * grads[k] / (np.sqrt(self.adagrad_h[k]) + self.epsilon)

class RMSprop:
    """RMSpropによる最適化クラス"""
    def __init__(self, learning_rate, beta_2, epsilon):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.beta_2 = beta_2
        self.epsilon = epsilon
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.adagrad_h = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[k] = self.beta_2 * self.adagrad_h[k] + (1 - self.beta_2) * grads[k] ** 2
            # パラメータ更新量 = 学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
            params[k] -= self.learning_rate * grads[k] / (np.sqrt(self.adagrad_h[k]) + self.epsilon)

class Adam:
    """Adamによる最適化クラス"""
    def __init__(self, learning_rate, beta_1, beta_2, epsilon, 
                 bias_correction=False, bias_max_iter=5000):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction  # バイアス補正の有無
        self.bias_max_iter = bias_max_iter  # バイアス補正を打ち切るイテレーション数
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.momentum_v = {k: np.zeros_like(params[k]) for k in params.keys()}
        self.adagrad_h = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均momentum_v = beta_1 * 更新前のmomentum_v - (1 - beta_1) * 勾配grads
            self.momentum_v[k] = self.beta_1 * self.momentum_v[k] + (1 - self.beta_1) * grads[k]
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[k] = self.beta_2 * self.adagrad_h[k] + (1 - self.beta_2) * grads[k] ** 2
            # バイアス補正 (最初のbias_max_iter回のみ実施)
            if self.bias_correction and i_iter < self.bias_max_iter:
                momentum_v_hat = self.momentum_v[k] / (1 - self.beta_1 ** (i_iter+1))
                adagrad_h_hat = self.adagrad_h[k] / (1 - self.beta_2 ** (i_iter+1))
            else:
                momentum_v_hat = self.momentum_v[k]
                adagrad_h_hat = self.adagrad_h[k]
            # パラメータ更新量 = 学習率learning_rate * momentum_v / (sqrt(adagrad_h)+epsilon)
            params[k] -= self.learning_rate * momentum_v_hat / (np.sqrt(adagrad_h_hat) + self.epsilon)

class AdamW:
    """AdamWによる最適化クラス"""
    def __init__(self, learning_rate, beta_1, beta_2, epsilon,
                 bias_correction=False, bias_max_iter=5000,
                 weight_decay_lambda=0.0001):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction  # バイアス補正の有無
        self.bias_max_iter = bias_max_iter  # バイアス補正を打ち切るイテレーション数
        self.weight_decay_lambda = weight_decay_lambda  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.momentum_v = {k: np.zeros_like(params[k]) for k in params.keys()}
        self.adagrad_h = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均momentum_v = beta_1 * 更新前のmomentum_v - (1 - beta_1) * 勾配grads
            self.momentum_v[k] = self.beta_1 * self.momentum_v[k] + (1 - self.beta_1) * grads[k]
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[k] = self.beta_2 * self.adagrad_h[k] + (1 - self.beta_2) * grads[k] ** 2
            # バイアス補正 (最初の1000回のみ実施)
            if self.bias_correction and i_iter < self.bias_max_iter:
                momentum_v_hat = self.momentum_v[k] / (1 - self.beta_1 ** (i_iter+1))
                adagrad_h_hat = self.adagrad_h[k] / (1 - self.beta_2 ** (i_iter+1))
            else:
                momentum_v_hat = self.momentum_v[k]
                adagrad_h_hat = self.adagrad_h[k]
            # パラメータ更新量 = 学習率learning_rate * momentum_v / (sqrt(adagrad_h)+epsilon)
            update = self.learning_rate * momentum_v_hat / (np.sqrt(adagrad_h_hat) + self.epsilon)
            # 重みパラメータのみ、パラメータ更新量に正則化項 (lr * weight_decay_lambda * 更新前のパラメータ)を加算
            if k == 'W':
                update += self.learning_rate * self.weight_decay_lambda * params[k]
            # パラメータ更新
            params[k] -= update