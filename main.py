# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from random import random

import numpy as np
from numpy import random


def Adaline(name):
    bias = np.random()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    d_size = 1000
    data = np.empty(d_size, dtype=object)
    random.seed(10)
    for i in range(d_size):
        data[i] = (random.randint(-100, 100, ) / 100, random.randint(-100, 100) / 100)

    train = np.empty(d_size, dtype=object)
    for i in range(d_size):
        if data[i][1] > 0.5 and data[i][0] > 0.5:
            train[i] = [data[i], 1]
        else:
            train[i] = [data[i], -1]
    print(train)