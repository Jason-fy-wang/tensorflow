import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

model_path = os.path.join(os.curdir, "model","linear")

tranX = np.linspace(-1, 1, 100)
tranY = 2 * tranX + np.random.randn(*tranX.shape) * 0.3

X = tf.convert_to_tensor(tranX, dtype=tf.float32)
Y = tf.convert_to_tensor(tranY, dtype=tf.float32)

class linear_model(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.W = tf.Variable(tf.random.normal([1]), name="weight")
        self.b = tf.Variable(tf.zeros([1]), name="bias")
    
    @tf.function
    def __call__(self, x):
        return tf.multiply(x,self.W) + self.b

linear_mode = linear_model()

def loss_compute(expect, predication):
    loss = tf.reduce_mean(tf.square(expect - predication))
    return loss

learnrate = 0.1
optimizer = tf.optimizers.SGD(learnrate)

epochs = 200
display_step = 2

# record train process
loss_history = []
weight_history = []
bias_history = []

platdata = {"batchsize": [], "loss": []}

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predict = linear_mode(X)
        loss = loss_compute(Y, predict)
        gradients = tape.gradient(loss, [linear_mode.W,linear_mode.b])
        optimizer.apply_gradients(zip(gradients,[linear_mode.W, linear_mode.b]))
        loss_history.append(loss.numpy())
        weight_history.append(linear_mode.W.numpy()[0])
        bias_history.append(linear_mode.b.numpy()[0])
        platdata["batchsize"].append(epoch)
        platdata["loss"].append(loss)

    if epoch % display_step == 0:
        print(f'Epoch {epoch}, loss : {loss.numpy()}, w= {linear_mode.W.numpy()}, b = {linear_mode.b.numpy()}')


tf.saved_model.save(linear_mode, model_path, signatures=linear_mode.__call__.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.float32)))

print("*" * 50)
print("finished")
print(f"final weight: {linear_mode.W.numpy()} (theory: 2.0), final bias: {linear_mode.b.numpy()}, (theory: 0.0)")

def make_predictions(val):
    x_tensor = tf.convert_to_tensor(val, dtype=tf.float32)
    preditions = linear_mode(x_tensor)
    return preditions.numpy()

def evaluate_mode():
    testx = np.linspace(-1,1,50)
    testy_expect = 2 * testx
    testy_pred = make_predictions(testx)

    mse = np.mean((testy_expect - testy_pred)**2)
    mae = np.mean(np.abs(testy_expect - testy_pred))
    r2 = 1 - (np.sum((testy_expect - testy_pred) ** 2) / np.sum((testy_expect - np.mean(testy_expect))**2) )
    print(f"均方误差MSE: {mse:.6f}, 平均绝对误差MAE: {mae:.6f}, 决定系数: {r2:.6f}")

evaluate_mode()

def show_data():
    fig,((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2,figsize=(15,20))
    
    # 1 原始数据和 拟合直线
    x = np.linspace(-1.2, 1.2, 100)
    expect_y = 2 * x
    predict_y = make_predictions(x)

    ax1.plot(tranX, tranY, "ro", label="train Data")
    ax1.plot(x, predict_y, "b-", linewidth=2, label=f"train line(y={linear_mode.W.numpy()[0]}x + {linear_mode.b.numpy()[0]})")
    ax1.plot(x, expect_y, "g--", linewidth=2, label="theory line(y=2x)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("result")
    ax1.legend()
    ax1.grid(True)

    # 2. change of loss
    ax2.plot(loss_history, "b-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_xlabel("Loss")
    ax2.set_title("loss change")
    ax2.grid(True)

    # 3. change of weight
    ax3.plot(weight_history, "r-", linewidth=2, label="weight")
    ax3.axhline(y=2.0, color='g', linestyle="--", label="theory")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("wight")
    ax3.set_title("change to weight")
    ax3.legend()
    ax3.grid(True)

    # 4. change of bias
    ax4.plot(bias_history, "m-", linewidth=2, label="bias")
    ax4.axhline(y=0.0, color="g", linestyle="--", label="theory")
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("bias")
    ax4.set_title("change of bias")
    ax4.legend()
    ax4.grid(True)
    plt.show()


show_data()




