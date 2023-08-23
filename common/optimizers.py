import numpy as np
from abc import ABCMeta, abstractmethod

class BaseOptimizer(metaclass=ABCMeta):
    """最適化クラスの継承用インタフェース"""
    def __init__(self):
        """クラスの初期化"""
        raise NotImplementedError()

    @abstractmethod
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        raise NotImplementedError()

    @abstractmethod
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        raise NotImplementedError()

class SGD(BaseOptimizer):
    """SGDによる最適化クラス"""
    def __init__(self, learning_rate=0.01,
                 weight_decay=0):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        pass
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            params[k] -= self.learning_rate * grads[k]

class Momentum(BaseOptimizer):
    """モーメンタムによる最適化クラス"""
    def __init__(self, learning_rate=0.1, momentum=0.9,
                 weight_decay=0):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        momentum : float
            勾配移動平均の減衰率ハイパーパラメータ
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.ms = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均ms = momentum * 更新前のms - 学習率learning_rate * 勾配grads
            self.ms[k] = self.momentum * self.ms[k]- self.learning_rate * grads[k]
            # パラメータ更新量 = ms
            params[k] += self.ms[k]

class AdaGrad(BaseOptimizer):
    """AdaGradによる最適化クラス"""
    def __init__(self, learning_rate=0.001, epsilon=1e-7,
                 weight_decay=0):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        epsilon : float
            ゼロ除算によるエラーを防ぐハイパーパラメータ
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.vs = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 過去の勾配2乗和vs = 更新前のvs + 勾配gradsの2乗
            self.vs[k] = self.vs[k] + grads[k] ** 2
            # パラメータ更新量 = -学習率learning_rate * 勾配grads / (sqrt(vs)+epsilon)
            params[k] -= self.learning_rate * grads[k] / (np.sqrt(self.vs[k]) + self.epsilon)

class RMSprop(BaseOptimizer):
    """RMSpropによる最適化クラス"""
    def __init__(self, learning_rate=0.01, beta_2=0.99, epsilon=1e-8,
                 weight_decay=0):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        beta_2 : float
            過去の勾配2乗和の減衰率ハイパーパラメータ
        epsilon : float
            ゼロ除算によるエラーを防ぐハイパーパラメータ
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.vs = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 過去の勾配2乗和vs = beta_2 * 更新前のvs + (1 - beta_2) * 勾配gradsの2乗
            self.vs[k] = self.beta_2 * self.vs[k] + (1 - self.beta_2) * grads[k] ** 2
            # パラメータ更新量 = 学習率learning_rate * 勾配grads / (sqrt(vs)+epsilon)
            params[k] -= self.learning_rate * grads[k] / (np.sqrt(self.vs[k]) + self.epsilon)

class Adam(BaseOptimizer):
    """Adamによる最適化クラス"""
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 bias_correction=False, bias_max_iter=5000,
                 weight_decay=0):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        beta_1 : float
            勾配移動平均の減衰率ハイパーパラメータ
        beta_2 : float
            過去の勾配2乗和の減衰率ハイパーパラメータ
        epsilon : float
            ゼロ除算によるエラーを防ぐハイパーパラメータ
        bias_correction : bool
            バイアス補正の有無
        bias_max_iter : int
            バイアス補正を打ち切るイテレーション数
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction  # バイアス補正の有無
        self.bias_max_iter = bias_max_iter  # バイアス補正を打ち切るイテレーション数
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.ms = {k: np.zeros_like(params[k]) for k in params.keys()}
        self.vs = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均ms = beta_1 * 更新前の - (1 - beta_1) * 勾配grads
            self.ms[k] = self.beta_1 * self.ms[k] + (1 - self.beta_1) * grads[k]
            # 過去の勾配2乗和vs = beta_2 * 更新前のvs + (1 - beta_2) * 勾配gradsの2乗
            self.vs[k] = self.beta_2 * self.vs[k] + (1 - self.beta_2) * grads[k] ** 2
            # バイアス補正 (最初のbias_max_iter回のみ実施)
            if self.bias_correction and i_iter < self.bias_max_iter:
                ms_hat = self.ms[k] / (1 - self.beta_1 ** (i_iter+1))
                vs_hat = self.vs[k] / (1 - self.beta_2 ** (i_iter+1))
            else:
                ms_hat = self.ms[k]
                vs_hat = self.vs[k]
            # パラメータ更新量 = 学習率learning_rate * ms / (sqrt(vs)+epsilon)
            params[k] -= self.learning_rate * ms_hat / (np.sqrt(vs_hat) + self.epsilon)

class AdamW(BaseOptimizer):
    """AdamWによる最適化クラス"""
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, 
                 bias_correction=False, bias_max_iter=5000,
                 weight_decay=0.004):
        """
        クラスの初期化

        Parameters
        ----------
        learning_rate : float
            学習率
        beta_1 : float
            勾配移動平均の減衰率ハイパーパラメータ
        beta_2 : float
            過去の勾配2乗和の減衰率ハイパーパラメータ
        epsilon : float
            ゼロ除算によるエラーを防ぐハイパーパラメータ
        bias_correction : bool
            バイアス補正の有無
        bias_max_iter : int
            バイアス補正を打ち切るイテレーション数
        weight_decay : float
            Weight decayの正則化効果の強さを表すハイパーパラメータ
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction  # バイアス補正の有無
        self.bias_max_iter = bias_max_iter  # バイアス補正を打ち切るイテレーション数
        self.weight_decay = weight_decay  # Weight decayの正則化効果の強さを表すハイパーパラメータ
    
    def initialize_opt_params(self, params):
        """最適化で使用する変数の初期化"""
        self.ms = {k: np.zeros_like(params[k]) for k in params.keys()}
        self.vs = {k: np.zeros_like(params[k]) for k in params.keys()}
    
    def update(self, params, grads, i_iter):
        """パラメータの更新"""
        for k in params.keys():  # パラメータごとに更新
            # 勾配移動平均ms = beta_1 * 更新前のms - (1 - beta_1) * 勾配grads
            self.ms[k] = self.beta_1 * self.ms[k] + (1 - self.beta_1) * grads[k]
            # 過去の勾配2乗和vs = beta_2 * 更新前のvs + (1 - beta_2) * 勾配gradsの2乗
            self.vs[k] = self.beta_2 * self.vs[k] + (1 - self.beta_2) * grads[k] ** 2
            # バイアス補正 (最初の1000回のみ実施)
            if self.bias_correction and i_iter < self.bias_max_iter:
                ms_hat = self.ms[k] / (1 - self.beta_1 ** (i_iter+1))
                vs_hat = self.vs[k] / (1 - self.beta_2 ** (i_iter+1))
            else:
                ms_hat = self.ms[k]
                vs_hat = self.vs[k]
            # パラメータ更新量 = 学習率learning_rate * ms / (sqrt(vs)+epsilon)
            update = self.learning_rate * ms_hat / (np.sqrt(vs_hat) + self.epsilon)
            # 重みパラメータのみ、パラメータ更新量に正則化項 (lr * weight_decay * 更新前のパラメータ)を加算
            if k == 'W':
                update += self.learning_rate * self.weight_decay * params[k]
            # パラメータ更新
            params[k] -= update