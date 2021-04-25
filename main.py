import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.AdalineAlgo import AdalineAlgo


def partA():
    print("\nPart A\n")
    d_size = 1000
    data = np.empty((d_size, 2), dtype=object)
    random.seed(10)
    for i in range(d_size):
        data[i, 0] = (random.randint(-100, 100) / 100)
        data[i, 1] = (random.randint(-100, 100) / 100)

    train = np.zeros(d_size)
    for i in range(d_size):
        if data[i][1] > 0.5 and data[i][0] > 0.5:
            train[i] = 1
        else:
            train[i] = -1

    X = data.astype(np.float64)  # test
    y = train.astype(np.float64)  # train
    # print("[X, y], output: ", [i for i in zip(data, train)])

    # create a Adaline classifier and train on our data
    # Learning rate: 1/100
    print("Learning rate: 1/100\n")
    classifier = AdalineAlgo(rate=0.01, niter=15).fit(X, y)
    print("score: ", classifier.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.01')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.01')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    # create a Adaline classifier and train on our data
    print("\nLearning rate: 1/10,000\n")
    classifier2 = AdalineAlgo(rate=0.0001, niter=15).fit(X, y)
    print("score: ", classifier2.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier2.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier2)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.0001')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.show()

    plt.plot(range(1, len(classifier2.cost_) + 1), classifier2.cost_)
    plt.title('(Part A) Adaline Algorithm - Learning rate 0.0001')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend(loc='upper left')
    plt.show()


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


def partB():
    print("\nPart B\n")
    d_size = 1000
    data = np.empty((d_size, 2), dtype=object)
    random.seed(10)
    for i in range(d_size):
        data[i, 0] = (random.randint(-100, 100) / 100)
        data[i, 1] = (random.randint(-100, 100) / 100)

    train = np.zeros(d_size)
    for i in range(d_size):
        if 0.5 <= (data[i][1] ** 2 + data[i][0] ** 2) <= 0.75:
            train[i] = 1
        else:
            train[i] = -1

    X = data.astype(np.float64)  # test
    y = train.astype(np.float64)  # train
    print("[X, y], output: ", [i for i in zip(data, train)])


if __name__ == '__main__':
    partA()
    # partB()
