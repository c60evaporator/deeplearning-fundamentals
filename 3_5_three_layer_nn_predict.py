import numpy as np

def init_network():
    """
    パラメータの初期化 (参考値として適当な値を入れている)
    """
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],  # 1層目の重みパラメータ
                              [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])  # 1層目のバイアスパラメータ
    network['W2'] = np.array([[0.1, 0.4], # 2層目の重みパラメータ
                              [0.2, 0.5],
                              [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])  # 2層目のバイアスパラメータ
    network['W3'] = np.array([[0.1, 0.3], # 3層目の重みパラメータ
                              [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])  # 3層目のバイアスパラメータ
    return network

def sigmoid(x):
    """
    シグモイド関数を計算
    """
    return 1 / (1 + np.exp(-x))

def forward_middle(z_prev, W, b):
    """
    中間層の順伝播計算
    """
    a = np.dot(z_prev, W) + b
    z = sigmoid(a)
    return z

def softmax(a):
    """
    ソフトマックス関数を計算
    """
    c = np.max(a)
    exp_a = np.exp(a - c)   # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def forward_last_classification(z_prev, W, b):
    """
    出力層の順伝播計算
    """
    a = np.dot(z_prev, W) + b
    y = softmax(a)
    return y

def forward_all(network, x):
    """
    順伝播を全て計算
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    z1 = forward_middle(x, W1, b1)
    z2 = forward_middle(z1, W2, b2)
    z3 = forward_last_classification(z2, W3, b3)
    return z3

network = init_network()  # パラメータの初期化
x = np.array([1.0, 0.5])  # 入力データ
y = forward_all(network, x)  # 順伝播を全て計算
print(y)  # 出力 [0.40625907 0.59374093]