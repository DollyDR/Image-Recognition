import tensorflow as tf
import numpy as np

rng = np.random.default_rng(2021)

class MyModel(tf.keras.Model):
  def __init__(self, output_classes, droput_rate = 0.3):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding='SAME' ,kernel_initializer='glorot_uniform', activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding='SAME', kernel_initializer='glorot_uniform', activation='relu')
    self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
    self.conv3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding='SAME', kernel_initializer='glorot_uniform', activation='relu')
    self.conv4 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding='SAME', kernel_initializer='glorot_uniform', activation='relu')
    self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
    self.flatten = tf.keras.layers.Flatten()
    self.dropout = tf.keras.layers.Dropout(droput_rate)
    self.d1 = tf.keras.layers.Dense(units = 512, kernel_initializer='glorot_uniform', activation='relu')
    self.d2 = tf.keras.layers.Dense(units = output_classes, kernel_initializer='glorot_uniform')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.max_pool1(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.max_pool2(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.d1(x)
    return self.d2(x)