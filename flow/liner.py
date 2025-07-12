import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tranX = np.linspace(-1, 1, 100)
tranY = 2 * tranX + np.random.randn(*tranX.shape) * 0.3

plt.plot(tranX, tranY, "ro", label="Original Data")
plt.legend()
plt.show()










