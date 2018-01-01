#!/usr/bin/python
# -*- coding: UTF-8 -*-#

import numpy as np
import urllib
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

# download the file
raw_data = urllib.urlopen(url)
#load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:, 0:7]
y = dataset[:, 8]

# 使用该数据集为例子 我们将特征矩阵作为X，将目标变量作为y
# 数据归一化

from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)
standardized_X = preprocessing.scale(X)

# 特征选择 特征选择是一个很需要创造力的过程，更多的是依赖于直觉和专业知识，并且有很多
# 现成的算法可以进行特征选择

# 下面是利用树算法来计算特征的信息量
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

print "----------------------------------------------------------"
# 算法的使用 scikit-learn实现了大部分机器学习的基础算法

## 1. 逻辑回归
## 大多数问题都可以归结为二元分类问题 这个算法的优点是可以给出数据所在类别的概率
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print model

# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
#显示主要的分类指标，返回每个类标签的精确、召回率及F1值

print(metrics.confusion_matrix(expected, predicted))
# 在机器学习领域，混淆矩阵（confusion matrix），又称为可能性表格或是错误矩阵。
# 它是一种特定的矩阵用来呈现算法性能的可视化效果，通常是监督学习（非监督学习，通常用匹配矩阵：matching
# matrix）。y_true: 是样本真实分类结果，y_pred: 是样本预测分类结果
print "--------------------------------------------------------"
## 2. 朴素贝叶斯算法
# 该方法的任务是还原训练样本数据的分布密度 其在多类别分类中有很好的效果
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X, y)
print model

# make predictions
expected = y
predicted = model.predict(X)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

print "--------------------------------------------------------"

## 3. K近邻算法
# k近邻算法常常被看作是分类算法的一部分 比如可以用它来评估特征  在特征选择上我们可以用到它
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)
print model

expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

print "--------------------------------------------------------"

## 4. 决策树
# 分类与回归树(Classification and Regression Trees ,CART)算法常用于特征含有类别信息的分类或者回归问题，这种方法非常适用于多分类情况。
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print model
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

print "--------------------------------------------------------"

## 5. 支持向量机
# SVM是非常流行的机器学习算法，主要用于分类问题，如同逻辑回归问题，它可以使用一对多的方法进行多类别的分类。
from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print model
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print "--------------------------------------------------------"


## 6. 如何优化算法参数
# 一项更加困难的任务是构建一个有效的方法用于选择正确的参数，
# 我们需要用搜索的方法来确定参数。scikit-learn提供了实现这一目标的函数。
import numpy as np
import warnings
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
warnings.filterwarnings("ignore")
# prepare a range of alpha values to test
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X, y)
print grid

# summarize the results of the grid search
print grid.best_score_
print grid.best_estimator_.alpha

print "--------------------------------------------------------"

## 7. 有时候随机从给定区间中选择参数是很有效的方法 然后根据这些参数来评估算法的效果进而选择最佳的那个
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
#prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(X, y)
print rsearch
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
