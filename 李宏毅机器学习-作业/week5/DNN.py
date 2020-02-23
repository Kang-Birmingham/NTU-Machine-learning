'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.utils import to_categorical


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='./data')

    return deal_data(x_train, y_train, x_test, y_test)


def deal_data(x_train, y_train, x_test, y_test):
    # reshape数据格式
    x_train = x_train.reshape(60000, 28*28)
    x_test = x_test.reshape(10000, 28*28)

    # 将数据格式进行转换
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 标准化
    x_train /= 255
    x_test /= 255

    print('trian samples:', x_train.shape[0])
    print('test samples:', x_test.shape[0])

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def dnn():
    model = Sequential()
    model.add(Dense(input_shape=(784, ), units=512, activation='sigmoid'))
    model.add(Dense(units=512, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.01),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    model = dnn()

    model.fit(x_train, y_train, batch_size=100, epochs=2)
    print('Train accuracy:', model.evaluate(x_train, y_train))
    print('Test accuracy:', model.evaluate(x_test, y_test))

