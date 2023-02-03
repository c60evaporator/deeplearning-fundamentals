import numpy as np

def calc_weight_init_std(prev_node_num, weight_init_std, activation_function):
        """
        重みパラメータ初期値の標準偏差を計算
        """
        # 自動計算する場合
        if weight_init_std == 'auto':
            # 活性化関数がSigmoidの時、Xavierの初期値を使用
            if activation_function == 'sigmoid':
                return np.sqrt(1.0 / prev_node_num)
            # 活性化関数がSigmoidの時、Xavierの初期値を使用
            elif activation_function == 'relu':
                return np.sqrt(2.0 / prev_node_num)
        # 固定値を指定する場合
        else:
            return weight_init_std
