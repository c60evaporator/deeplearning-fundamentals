import numpy as np

def forward_batch_normalization(Z_prev, gamma, beta, epsilon):
    """
    Batch Normalizationの順伝播計算
    """
    mu = np.mean(Z_prev, axis=0)  # バッチ平均
    Zc = Z_prev - mu  # バッチ平均との差分
    var = np.mean(Zc ** 2, axis=0)  # バッチ分散
    A = Zc / np.sqrt(var + epsilon)  # 標準化された前層出力 (epsilonはゼロ除算対策)
    Z = gamma * A + beta  # モーメンタムとの補正
    # 出力Zだけでなく逆伝播で使用する中間結果も一緒に出力
    return Z, A, var, Zc, mu