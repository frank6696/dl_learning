# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(arr):
    return 1.0/(1.0+np.exp(-arr))


# 梯度下降法
def gradient_descent(X, y, alpha, max_iter):
    J = np.arange(max_iter, dtype=float)  # loss function
    m,n = X.shape
    theta = np.ones((n,1),dtype=float)  # 初始化参数
    for i in range(max_iter):
        h = sigmoid(np.dot(X,theta.astype(float)))
        J[i] = -(np.sum(y*np.log(h)+(1-y)*np.log(1-h)))/m
        error = h - y
        grad = np.dot(X.T, error)
        theta = theta- alpha * grad
    plt.plot(range(max_iter), J)
    plt.show()


if __name__ == '__main__':
    # load data
    iris = pd.read_csv('data/iris.csv')
    iris.loc[iris['Species'] == 'setosa','Species'] = 1
    iris.loc[iris['Species'] == 'versicolor','Species'] = 1
    # print(iris.head())
    X = iris.drop(['Species','id'],axis=1)
    y = iris['Species']
    gradient_descent(X.values, y.values.reshape(y.shape[0],1), 0.1, 100)
