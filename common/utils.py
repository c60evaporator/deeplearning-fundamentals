import numpy as np

def calc_weight_init_std(prev_node_num, weight_init_std, activation_function):
    """
    重みパラメータ初期値の標準偏差を計算
    """
    # 自動計算する場合
    if weight_init_std == 'auto':
        # 活性化関数がSigmoid or Softmaxの時、Xavierの初期値を使用
        if activation_function in ['sigmoid', 'softmax']:
            return np.sqrt(1.0 / prev_node_num)
        # 活性化関数がSigmoidの時、Xavierの初期値を使用
        elif activation_function == 'relu':
            return np.sqrt(2.0 / prev_node_num)
    # 固定値を指定する場合
    else:
        return weight_init_std

def flatten(input_data):
    """
    画像(2次元以上)を1次元配列に変換
    (実際はバッチデータのデータ数も含めた3次元以上→2次元に変換される)
    """
    original_shape = input_data.shape
    return input_data.reshape(original_shape[0], -1)

def calc_flatten_shape(input_shape):
    """
    flatten後のデータ形状を計算 (バッチデータのデータ数は次元に含めない)
    """
    reshaped_shape = (np.prod(input_shape), )
    return reshaped_shape