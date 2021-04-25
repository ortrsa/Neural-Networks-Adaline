import numpy as np


class AdalineAlgo:
    def __init__(self, rate=0.01, niter=15):
        self.learning_rate = rate
        self.niter = niter

        # Number of Incorrect classifications
        self.errors = []

        # Cost function
        self.cost_ = []

    def fit(self, X, y):
        """
        Fit training data.
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        :param X:
        :param y:
        :return:
        """
        row = X.shape[0]
        col = X.shape[1]

        #  add bias to X
        X_bias = np.ones((row, col + 1))
        X_bias[:, 1:] = X
        X = X_bias

        # weights - set size of weight Vector
        np.random.seed(1)
        self.weight = np.random.rand(col + 1)

        # training
        for _ in range(self.niter):
            """
            output: Common output
            y: Desired output
            """
            output = self.net_input(X)
            errors = y - output  # vector

            """
            We defined: X.T[0] = 1 
            this is the reason why the bias calc (= self.weight[0])
            will be equal to: 
            self.weight[0] += self.learning_rate * errors
            """
            self.weight += self.learning_rate * (X.T @ errors)
            # self.weight[1:] += self.learning_rate * (X.T @ errors)
            # self.weight[0] += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        Calculate net input.
        :param X:
        :return:
        """
        return X @ self.weight
        # return (X @ self.weight[1:]) + self.weight[0]

    def activation(self, X):
        """
        Compute linear activation.
        :param X:
        :return:
        """
        return self.net_input(X)

    def predict(self, X):
        """
        Return class label after unit step.
        :param X:
        :return:
        """
        # if x is list instead of np.array
        if type(X) is list:
            X = np.array(X)

        # add bias to x if he doesn't exist
        if len(X.T) != len(self.weight):
            X_bias = np.ones((X.shape[0], X.shape[1] + 1))
            X_bias[:, 1:] = X
            X = X_bias

        return np.where(self.activation(X) >= 0.0, 1, -1)
        # return np.where(self.activation(X) >= 0.0, 1, -1)

    def score(self, X, y):
        """
        Model score is calculated based on comparison of
        expected value and predicted value.
        :param X:
        :param y:
        :return:
        """
        wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        self.score_ = (len(X) - wrong_prediction) / len(X)
        # print("wrong_prediction ", wrong_prediction)
        return self.score_
