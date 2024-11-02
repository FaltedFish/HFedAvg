import numpy as np


def channel_demo(channel_gain, emission_coefficient, input):
    return channel_gain * emission_coefficient * input + np.random.normal(0, 1e-11)


if __name__ == "__main__":
    input = np.random.normal(0, 1,(2,2))
    print(input)
    channel_gain = np.random.normal(0, 1)
    b = 1
    print(channel_demo(channel_gain, b, input))
