import numpy as np

def softmax_loss_backward(Y, T):
    """
    Softmax-with-Lossレイヤの逆伝播
    """
    batch_size = T.shape[0]
    dA = (Y - T) / batch_size
    return dA

def affine_backward_bias(dA):
    """
    Affineレイヤの逆伝播 (バイアスパラメータbの偏微分を計算)
    """
    db = np.sum(dA, axis=0)  # 入力dAを縦方向に合計
    return db

def affine_backward_weight(dA, Z_prev):
    """
    Affineレイヤの逆伝播 (重みパラメータWの偏微分を計算)
    """
    dW = np.dot(Z_prev.T, dA)  # 前層出力Z_prevの転置とdAの行列積
    return dW

def affine_backward_zprev(dA, W):
    """
    Affineレイヤの逆伝播 (前層出力Z_prevの偏微分を計算)
    """
    dZ_prev = np.dot(dA, W.T)  # dAと重みパラメータWの転置の積
    return dZ_prev

def relu_backward(dZ, A):
    """
    ReLUレイヤの逆伝播
    """
    mask = (A <= 0)  # a ≦ 0を判定する行列
    dA = dZ.copy()  # dZを一旦入力
    dA[mask] = 0  # a ≦ 0の成分のみを0に置き換え
    return dA

def sigmoid_backward(dZ, Z):
    """
    Sigmoidレイヤの逆伝播
    """
    dA = Z * (1.0 - Z) * dZ  # 各成分の掛け合わせでdAを計算
    return dA