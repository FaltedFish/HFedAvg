import numpy as np


def get_e():
    return np.random.normal(0, 1e-11)


def single_channel(signal, channel_gain, emission_coefficient, e):
    """
    计算最简单的信道模型
    参数：
    - signal: 发射端发出的信号。
    - channel_gain: 信号从发射端到接收端的信道增益。
    - emission_coefficient: 表示发射系数。
    """
    return channel_gain * emission_coefficient * signal + e


def received_by_relay(signal, channel_gain, emission_coefficient, other_relay_channel_gain,
                      other_relay_signal_emission_coefficient, other_relay_signal, e):
    """
        计算中继 r 中收到空中计算组 g 的叠加信号。

        参数:
        - signal: 空中计算组 g聚合后的信号。
        - channel_gain: 信号从发射机到中继的信道增益。
        - emission_coefficient: 表示从另一个中继 到该中继的信道增益。
        - other_relay_channel_gain: 信号从另一个中继到当前中继的信道增益。
        - other_relay_signal_emission_coefficient: 表示簇外某中继器的发射系数。
        - other_relay_signal: 该中继器发射的信号。
        - e: 中继 r 的加性高斯白噪声

        返回:
        - 中继 r 中收到空中计算组 g 的叠加信号。
        """
    return (signal * channel_gain * emission_coefficient + other_relay_signal *
            other_relay_channel_gain * other_relay_signal_emission_coefficient + e)
