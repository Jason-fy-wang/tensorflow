import tensorflow as tf
import os

model_path = os.path.join(os.curdir, "model","linear")


model = tf.saved_model.load(model_path)

print(f"prediction: {model(2.0)}")
