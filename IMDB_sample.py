from keras.datasets import imdb
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimensions=10000):
  results = np.zeros((len(sequences), dimensions))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
  num_words=10000)

x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype("float32")

x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype("float32")

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

# history_dict = history.history
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]

# epochs = range(1, 21)

# plt.plot(epochs, loss_values, "bo", label="Training Loss")
# plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.show()