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