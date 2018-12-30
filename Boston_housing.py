from keras.datasets import boston_housing
from keras import models
from keras import layers

import numpy as np

def build_model(train_data):
  # Typical set up for scalar regression

  model = models.Sequential()
  model.add(layers.Dense(64, activation="relu",
    input_shape=(train_data.shape[1],)))
  # Last layer has no activation -> Linear layer
  model.add(layers.Dense(1))

  # mse: means square error loss function
  # the square of the difference between the predictions and the targets
  # mae: mean absolute error:
  # the absolute value of the difference between the predictions and the targets

  model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

(train_data, train_targets), (test_data, test_targets) = 
  boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean

# standard devition
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std