from random import seed

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from src.AdalineAlgo import AdalineAlgo


def creatData(d_size, Part):
    data = np.empty((d_size, 2), dtype=object)
    random.seed(10)
    for i in range(d_size):
        data[i, 0] = (random.randint(-100, 100) / 100)
        data[i, 1] = (random.randint(-100, 100) / 100)

    train = np.zeros(d_size)

    if Part == "A":
        for i in range(d_size):
            if data[i][1] > 0.5 and data[i][0] > 0.5:
                train[i] = 1
            else:
                train[i] = -1

    if Part == "B":
        for i in range(d_size):
            if 0.5 <= (data[i][1] ** 2 + data[i][0] ** 2) <= 0.75:
                train[i] = 1
            else:
                train[i] = -1

    X = data.astype(np.float64)  # test
    y = train.astype(np.float64)  # train

    # print("[X, y], output: ", [i for i in zip(data, train)])

    return X, y


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 1], y=X[y == cl, 0],
                    alpha=0.9, c=np.atleast_2d(cmap(idx)),
                    marker=markers[idx], label=cl)


def partA():
    print("\nPart A\n")
    d_size = 1000
    X, y = creatData(d_size, "A")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # learning rate = 0.01
    classifier = AdalineAlgo(0.01, 10).fit(X, y)
    ax[0].plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('(Part A) Adaline Algorithm - Learning rate 0.01')

    # learning rate = 0.0001
    classifier2 = AdalineAlgo(0.0001, 10).fit(X, y)
    ax[1].plot(range(1, len(classifier2.cost_) + 1), classifier2.cost_, marker='o')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('(Part A) Adaline Algorithm - Learning rat 0.0001')
    plt.show()

    # create a Adaline classifier and train on our data
    # Learning rate: 1/100
    print("Learning rate: 1/100\n")
    print("score: ", classifier.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier.cost_).min())
    print("confusion_matrix: ", confusion_matrix(classifier.predict(X), y))

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.01')
    plt.legend(loc='upper left')
    plt.show()

    # create a Adaline classifier and train on our data
    # learning rate = 0.0001
    print("\nLearning rate: 1/10,000\n")
    print("score: ", classifier2.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier2.cost_).min())
    print("confusion_matrix: ", confusion_matrix(classifier2.predict(X), y))

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier2)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.0001')
    plt.legend(loc='upper left')
    plt.show()


def partB():
    print("\nPart B\n")

    # data = 1,000
    X1, y1 = creatData(1000, "B")
    classifier = AdalineAlgo(0.0001, 50).fit(X1, y1)

    # data = 100,000
    X2, y2 = creatData(100000, "B")
    classifier2 = AdalineAlgo(0.0001, 50).fit(X2, y2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # learning rate = 0.0001 || d_type = 1,000
    ax[0].plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('(Part B) Adaline Algorithm - d_type 1,000')

    # learning rate = 0.0001 || d_type = 100,000
    ax[1].plot(range(1, len(classifier2.cost_) + 1), classifier2.cost_, marker='o')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('(Part B) Adaline Algorithm - d_type 100,000')
    plt.show()

    # create a Adaline classifier and train on our data
    # learning rate = 0.0001 || d_type = 1,000
    print("d_type: 1,000\n")
    print("score: ", classifier.score(X1, y1) * 100, "%")
    print("cost: ", np.array(classifier.cost_).min())
    print("confusion_matrix: ", confusion_matrix(classifier.predict(X1), y1))

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X1, y1, classifier=classifier)
    plt.title('(Part B) Adaline Algorithm - d_type 1,000')
    plt.legend(loc='upper left')
    plt.show()

    # create a Adaline classifier and train on our data
    # learning rate = 0.0001 || d_type = 100,000
    print("\nd_type: 10,000\n")
    print("score: ", classifier2.score(X2, y2) * 100, "%")
    print("cost: ", np.array(classifier2.cost_).min())
    print("confusion_matrix: ", confusion_matrix(classifier2.predict(X2), y2))

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X2, y2, classifier=classifier2)
    plt.title('(Part B) Adaline Algorithm - d_type 100,000')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    partA()
    partB()
