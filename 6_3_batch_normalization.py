#%% 6.1の内容にBatch Normalizationを適用し、中間層の出力分布をプロット
import numpy as np
import matplotlib.pyplot as plt
from common.activation_functions import sigmoid, relu
from common.forward_advanced import forward_batch_normalization

input_data = np.random.randn(1000, 100)  # 1000個の入力データ=バッチサイズ(標準正規分布から生成)
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層

# 活性化関数を変えて実験する
for activation_function in ['sigmoid', 'relu']:

    # 重み初期値生成時の標準偏差を変えて実験する
    weight_init_stds = {'Std=1.0': 1.0,
                        'Std=0.01': 0.01,
                        'Xavier': np.sqrt(1.0 / node_num), 
                        'He': np.sqrt(2.0 / node_num)}
    
    # 結果描画用のキャンバス
    fig, axes = plt.subplots(4, hidden_layer_size, figsize=(3*hidden_layer_size, 4*4))
    fig.suptitle(f'activation_function = {activation_function}', fontsize=32)

    for i_std, (std_label, weight_init_std) in enumerate(weight_init_stds.items()):
        activations = {}  # 各層出力格納用
        # 入力データ
        x = input_data
        # 全層ループ
        for i in range(hidden_layer_size):
            # 前層出力を入力とする
            if i != 0:
                x = activations[i-1]

            # 重みパラメータの初期化 (標準偏差weight_init_stdの正規分布)
            w = np.random.randn(node_num, node_num) * weight_init_std
            # Affineレイヤ
            a = np.dot(x, w)
            # Batch Normalization
            a,_ ,_ ,_ ,_ = forward_batch_normalization(a, gamma=1, beta=0, epsilon=1e-6)
            # 活性化関数(Sigmoid)
            if activation_function == 'sigmoid':
                z = sigmoid(a)
            # 活性化関数 (ReLU)
            elif activation_function == 'relu':
                z = relu(a)
            # 各層の出力を格納
            activations[i] = z

        # 層ごとにヒストグラムを描画
        for i, a in activations.items():
            ax = axes[i_std][i]
            ax.set_title(str(i+1) + "-layer")
            if i == 0:
                ax.set_ylabel(std_label, fontsize=18)
            else:
                ax.set_yticks([])
            ax.hist(a.flatten(), 30, range=(0,1))
    plt.show()
# %%
