# Author: [Seyedsaman Emami](https://github.com/samanemami)

# Licence: GNU General Public License v3.0

# Abouth this script

# In this notebook, I am working with a well-known image processing dataset in the computer vision field, MNIST.

# The model is a Convolutional Neural Network - CNN with the following layers. The optimizer is Stochastic Gradient Descent.

# A novelty of my script is that I am comparing the model performance with an in_train accuracy.

# To follow my developments, meet me on GitHub(https://github.com/samanemami)

import warnings
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

warnings.simplefilter("ignore")
random_state = 1

np.random.seed(random_state)
set_seed(random_state)


np.set_printoptions(precision=5, suppress=True)


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


class mnist_():

    def __init__(self, nb_classes=10,
                 size=28,
                 channels=1) -> None:

        self.nb_classes = nb_classes
        self.size = size
        self.channels = channels

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def train(self):
        X_train = prepare_data(self.X_train, self.size, self.channels)
        X_test = prepare_data(self.X_test, self.size, self.channels)
        y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
        return X_train, X_test, y_train, y_test

    def in_train(self, random_state):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size=0.5, random_state=random_state)
        X_train = prepare_data(X_train, self.size, self.channels)
        X_test = prepare_data(X_test, self.size, self.channels)
        y_train = np_utils.to_categorical(y_train, self.nb_classes)
        y_test = np_utils.to_categorical(y_test, self.nb_classes)
        return X_train, X_test, y_train, y_test


class cnn():
    def __init__(self, size, channels) -> None:
        self.size = size
        self.channels = channels

    def fit(self, X_train, Y_train, X_test, Y_test):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(
            self.size, self.size, self.channels), kernel_initializer='he_uniform', padding='same'))
        convout1 = Activation('relu')
        model.add(convout1)
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',
                  kernel_initializer='he_uniform'))

        model.add(Dense(128, activation='relu',
                  kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, Y_train,
                  batch_size=128,
                  epochs=50,
                  verbose=0,
                  validation_data=(X_test, Y_test))

        score = model.evaluate(X_test,
                               Y_test,
                               verbose=0)

        return score[1]


X_train = mnist_().train()[0]
X_test = mnist_().train()[1]
Y_train = mnist_().train()[2]
Y_test = mnist_().train()[3]
model = cnn(28, 1)
overall = model.fit(X_train, Y_train, X_test, Y_test)

X_train = mnist_().in_train(random_state)[0]
X_test = mnist_().in_train(random_state)[1]
Y_train = mnist_().in_train(random_state)[2]
Y_test = mnist_().in_train(random_state)[3]
model = cnn(28, 1)
fold1 = model.fit(X_train, Y_train, X_test, Y_test)

X_test = mnist_().in_train(random_state)[0]
X_train = mnist_().in_train(random_state)[1]
Y_test = mnist_().in_train(random_state)[2]
Y_train = mnist_().in_train(random_state)[3]
model = cnn(28, 1)
fold2 = model.fit(X_train, Y_train, X_test, Y_test)


df = pd.DataFrame([overall, np.mean((fold1, fold2))],
                  index=['overall', 'in_train'])
df.columns = ['accuracy']
df.head()
