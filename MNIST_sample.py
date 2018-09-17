from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
network = models.Sequential()

# network have two Dense layers (fully connected)
# The second layer is a 10-way softmax layer, return an array of 10 probability scores (sum-
# ming to 1). Each score will be the probability that the current digit image belongs to
# one of our 10 digit classes

# This layer of network will only accept input tensor's shape: 28 * 28 (axis 0)
# (2D tensor), batch dimension is not specified (any value would be accepted)
# output is tensor that first dimension will be transformed into 512
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))

network.add(layers.Dense(10, activation="softmax"))
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# Preprocess data to change shape and scale values of pixels in images are in [0, 1] 

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_acc: ", test_acc)

# my_slice = train_images[10:100] # doesn't include 100
# print my_slice.shape

# my_slice = train_images[10:100, :, :] # equivalent to previous example
# print my_slice.shape

# my_slice = train_images[10:100, 1:28, 2:28]
# print my_slice.shape