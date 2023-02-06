import numpy as np

class SGD:
    """SGDによる最適化クラス"""
    def __init__(self, learning_rate):
        """クラスの初期化"""
        self.learning_rate = learning_rate
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        pass
    
    def update(self, params, grads):
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
    
    def update(self, params, grads):
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
    
    def update(self, params, grads):
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
    
    def update(self, params, grads):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[k] = self.beta_2 * self.adagrad_h[k] + (1 - self.beta_2) * grads[k] ** 2
            # パラメータ更新量 = 学習率learning_rate * 勾配grads / (sqrt(adagrad_h)+epsilon)
            params[k] -= self.learning_rate * grads[k] / (np.sqrt(self.adagrad_h[k]) + self.epsilon)

class Adam:
    """Adamによる最適化クラス"""
    def __init__(self, learning_rate, beta_1, beta_2, epsilon):
        """クラスの初期化"""
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.momentum_v = {k: np.zeros_like(params[k]) for k in params.keys()}
        self.adagrad_h = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均momentum_v = beta_1 * 更新前のmomentum_v - (1 - beta_1) * 勾配grads
            self.momentum_v[k] = self.beta_1 * self.momentum_v[k] + (1 - self.beta_1) * grads[k]
            # 過去の勾配2乗和adagrad_h = beta_2 * 更新前のadagrad_h + (1 - beta_2) * 勾配gradsの2乗
            self.adagrad_h[k] = self.beta_2 * self.adagrad_h[k] + (1 - self.beta_2) * grads[k] ** 2
            # パラメータ更新量 = 学習率learning_rate * momentum_v / (sqrt(adagrad_h)+epsilon)
            params[k] -= self.learning_rate * self.momentum_v[k] / (np.sqrt(self.adagrad_h[k]) + self.epsilon)