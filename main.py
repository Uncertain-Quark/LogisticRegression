# Date: 19th Oct, 2022
# Author: Aditya Kommineni
import keras.datasets.mnist
from logisticregression import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(y_train[:30])

    y_train = (y_train == 2).reshape(-1,1).astype(int)
    y_test = (y_test == 2).reshape(-1,1).astype(int)
    print(y_train[:30].T)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    print(x_test.shape)
    n_iters = 2000
    model = binarylogisticregression(0.0005, 784, 1, n_iters)
    w, l = model.train(x_train, y_train)
    plt.plot(l)
    plt.show()
    