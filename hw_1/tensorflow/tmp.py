import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Sequential


print(tf.__version__)
model = Sequential()
model.add(Flatten())
model.build((10, 5,5))
print(model.summary())