import numpy as np
import matplotlib.pyplot as plt

# 微分函数
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h


def numberical_diff2(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h) / (2*h))


# 计算 梯度
def numberical_gradient(f, x):
    h = 1e-4
    grad = np.zeros(x.size)

    for idx in range(x.size):
        tmp = x[idx]

        # f(x+h) val
        x[idx] = tmp + h
        fxh1 = f(x)

        # f(x-h) val
        x[idx] = tmp - h
        fxh2 = f(x)

        grad[idx] = (fxh1 + fxh2) / (2 * h)
        x[idx] = tmp
    return grad

