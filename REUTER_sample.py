from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
  num_words=10000)

def vectorize_sequences(sequences, dimensions=10000):
  results = np.zeros((len(sequences), dimensions))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

def to_one_hot(labels, dimensions=46):
  results = np.zeros((len(labels), dimensions))
  for i, label in enumerate(labels):
    results[i, label] = 1
  return results

# MARK: Use one-hot encoding

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

# model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
#   metrics=["accuracy"])

# model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512,
#   validation_data=(x_val, y_val))

# results = model.evaluate(x_test, one_hot_test_labels)

# history = model.fit(partial_x_train, partial_y_train,
#   epochs=20, batch_size=512, validation_data=(x_val, y_val))

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and Validation losss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.show()

# MARK: Use Integer Casting
y_train = np.array(train_labels)
y_test = np.array(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
  metrics=["acc"])

model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=128,
  validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print results
