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