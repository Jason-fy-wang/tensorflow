import numpy as np
import matplotlib.pyplot as plt

# 激活函数

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# 阶跃函数
def step(x):
    return np.array(x > 0, dtype=np.int64)


def relu(x):
    return np.maximum(0, x)


# softmax hanshu 
def softmax(x):
    expa = np.exp(x)
    exp_sum = np.sum(expa)
    return expa / exp_sum

def softmax_safe(x):
    c = np.max(x)
    exp_a = np.exp(a - c)  # 防止溢出
    exp_sum = np.sum(exp_a)
    return exp_a / exp_sum


def show_step():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def show_relu():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def show_sigmoid():
    x = np.arange(-8.0, 9.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()





if __name__ == '__main__':
    #show_step()
    #show_relu()
    show_sigmoid()
