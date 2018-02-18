# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


class LR:
    weight = None

    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.max_iter = max_iter

    def sigmoid(self, arr):
        return 1.0 / (1.0 + np.exp(-arr))

    # 梯度下降法
    def gradient_descent(self, X, y):
        J = np.arange(self.max_iter, dtype=float)  # loss function
        m, n = X.shape
        theta = np.ones((n, 1), dtype=float)  # 初始化参数
        for i in range(self.max_iter):
            h = self.sigmoid(np.dot(X, theta.astype(float)))
            J[i] = -(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))) / m
            error = h - y
            grad = np.dot(X.T, error)
            theta = theta - self.alpha * grad
        self.weight = theta
        # plt.plot(range(self.max_iter), J)
        # plt.show()

    def predict(self, X):
        h = self.sigmoid(np.dot(X, self.weight.astype(float)))
        h[h > 0.5] = 1
        h[h < 0.5] = 0
        return h[:, 0].astype(int)