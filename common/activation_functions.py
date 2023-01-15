import numpy as np

def step_function(X):
    """
    ステップ関数を計算
    """
    return np.array(X > 0, dtype=np.int)

def sigmoid(X):
    """
    シグモイド関数を計算
    """
    return 1 / (1 + np.exp(-X))

def relu(X):
    """
    ReLU関数を計算
    """
    return np.maximum(0, X)

def softmax(A):
    """
    ソフトマックス関数を計算
    """
    C = np.max(A, axis=-1, keepdims=True)  # バッチ処理のため、横方向に最大値計算＆次元を維持
    exp_A = np.exp(A - C)   # オーバーフロー対策
    sum_exp_A = np.sum(exp_A, axis=-1, keepdims=True)  # バッチ処理のため、横方向に合計計算＆次元を維持
    Y = exp_A / sum_exp_A
    return Y

def identity(X):
    """
    恒等関数を計算
    """
    return X