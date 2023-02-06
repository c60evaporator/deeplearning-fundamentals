import numpy as np

def backward_batch_normalization(dZ, gamma, batch_size, A, var, Zc):
    """
    Batch Normalizationの逆伝播計算
    """
    dA = gamma * dZ  # A偏微分 (逆伝播1)
    dgamma = np.sum(A * dZ, axis=0) # gamma偏微分 (逆伝播1)
    dbeta = np.sum(dZ, axis=0)  # beta偏微分 (逆伝播1)
    dvar = -0.5 * np.sum((dA * Zc) / (var * np.sqrt(var)), axis=0)  # 分散var偏微分 (逆伝播2)
    dZc = dA / np.sqrt(var) \
        + (2.0 / batch_size) * Zc * dvar # 平均との差Zc偏微分 (逆伝播3＆4)
    dmu = -np.sum(dZc, axis=0)  # 平均mu偏微分 (逆伝播5)
    dZ_prev = dZc + dmu / batch_size  # 前層出力Z_prev偏微分 (逆伝播6)

    # 逆伝播出力dZ_prevだけでなくパラメータの偏微分dgamma, dbetaも出力
    return dZ_prev, dgamma, dbeta