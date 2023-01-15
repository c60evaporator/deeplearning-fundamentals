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

def sigmoid(X):
    """
    シグモイド関数を計算
    """
    return 1 / (1 + np.exp(-X))

def forward_middle(Z_prev, W, b):
    """
    中間層の順伝播計算
    """
    A = np.dot(Z_prev, W) + b
    Z = sigmoid(A)
    return Z

def softmax(A):
    """
    ソフトマックス関数を計算
    """
    C = np.max(A, axis=-1, keepdims=True)  # バッチ処理のため、横方向に最大値計算＆次元を維持
    exp_A = np.exp(A - C)   # オーバーフロー対策
    sum_exp_A = np.sum(exp_A, axis=-1, keepdims=True)  # バッチ処理のため、横方向に合計計算＆次元を維持
    Y = exp_A / sum_exp_A
    return Y

def forward_last_classification(Z_prev, W, b):
    """
    出力層の順伝播計算
    """
    A = np.dot(Z_prev, W) + b
    Y = softmax(A)
    return Y

def forward_all(network, X):
    """
    順伝播を全て計算
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    Z1 = forward_middle(X, W1, b1)
    Z2 = forward_middle(Z1, W2, b2)
    Z3 = forward_last_classification(Z2, W3, b3)
    return Z3

network = init_network()  # パラメータの初期化
X = np.array([[1.0, 0.5], [1.1, 0.8], [0.8, 0.4]])  # 入力データ (3データ)
Y = forward_all(network, X)  # 順伝播を全て計算
print(Y)  # 出力 [[0.40625907 0.59374093] [0.40563195 0.59436805] [0.40671823 0.59328177]]