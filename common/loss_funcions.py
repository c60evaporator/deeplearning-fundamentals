import numpy as np
from common.activation_functions import softmax, identity, sigmoid, relu, step_function

def cross_entropy_error(Y, T):
    """
    交差エントロピー誤差を計算
    """
    # 交差エントロピー誤差を計算して返す
    batch_size = Y.shape[0]
    return -np.sum(T * np.log(Y)) / batch_size

def squared_error(Y, T):
    """
    2乗和誤差を計算
    """
    # 2乗和誤差を計算して返す
    batch_size = Y.shape[0]
    return 1.0/2.0 * np.sum(np.square(Y - T))

def numerical_gradient1(f, P):
    """
    勾配を計算
    """
    h = 1e-4  # 原書に記載された適切な微小変化量hの値として1e-4を採用
    grad = np.zeros_like(P)  # パラメータPと同じ形状のベクトルを生成
    # 変数ごと(列ごと)に偏微分を計算
    for idx in range(P.size):
        tmp_val = P[idx]
        P[idx] = tmp_val + h
        print(tmp_val)
        print(P[idx])
        fxh1 = f(P)
        
        P[idx] = tmp_val - h
        fxh2 = f(P)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        P[idx] = tmp_val
    return grad

def numerical_gradient(f, params, param_name, l):
    """
    勾配を計算

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or pd.ndarray
        Input data structure. Either a long-form collection of vectors that can be assigned to named variables or a wide-form dataset that will be internally reshaped.
    params : 
        パラメータ ()
    param_name : {'W', 'b'}
        パラメータ名 ('W': 重みパラメータ, 'b': バイアス)
    l : int
        何層目のパラメータか
    """
    P = param

    h = 1e-4  # 原書に記載された適切な微小変化量hの値として1e-4を採用
    grad = np.zeros_like(P)  # Pと同じ形状のベクトルor行列を生成
    P_ravel = np.ravel(P)  # Pが行列(重みパラメータ)の時、一旦ベクトルとして展開

    # パラメータごとに偏微分を計算
    for idx in range(P_ravel.size):
        # f(x+h)の計算
        h_matrix = np.zeros_like(P)  # 微小変化量に相当するベクトル(該当パラメータに相当する成分のみh、他成分は全て0)
        fxh1 = f(P + h_matrix)
        # f(x-h)の計算
        fxh2 = f(P - h_matrix)
        # 偏微分の計算
        grad[idx] = (fxh1 - fxh2) / (2*h)
    return grad