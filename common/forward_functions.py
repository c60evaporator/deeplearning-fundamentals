import numpy as np
from common.activation_functions import softmax, identity, sigmoid, relu, step_function

def forward_middle(Z_prev, W, b, activation_function='sigmoid', output_A=False):
    """
    中間層の順伝播計算
    """
    A = np.dot(Z_prev, W) + b
    if activation_function == 'sigmoid':
        Z = sigmoid(A)
    elif activation_function == 'relu':
        Z = relu(A)
    elif activation_function == 'step':
        Z = step_function(A)
    else:
        raise Exception('The activation_function should be "sigmoid", "relu", or "step"')
    # Zだけでなく中間結果Aも一緒に出力する場合 (5章の誤差逆伝播法で使用)
    if output_A:
        return Z, A
    # Zのみ出力する場合
    else:
        return Z

def forward_last_classification(Z_prev, W, b):
    """
    出力層の順伝播計算(分類)
    """
    A = np.dot(Z_prev, W) + b
    Y = softmax(A)
    return Y

def forward_last_regression(z_prev, W, b):
    """
    出力層の順伝播計算(回帰)
    """
    a = np.dot(z_prev, W) + b
    y = identity(a)
    return y