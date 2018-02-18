# coding=utf-8
from LR import LR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


if __name__ == '__main__':
    # load data
    iris = pd.read_csv('data/iris.csv')
    iris.loc[iris['Species'] == 'setosa','Species'] = 1
    iris.loc[iris['Species'] == 'versicolor','Species'] = 0
    # print(iris.head())
    X = iris.drop(['Species','id'],axis=1)
    y = iris['Species']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
    lr = LR(0.1, 100)
    lr.gradient_descent(X_train.values, y_train.values.reshape(y_train.shape[0], 1))
    y_pred = lr.predict(X_test)
    print(f1_score(y_test.values.astype(int),y_pred))
