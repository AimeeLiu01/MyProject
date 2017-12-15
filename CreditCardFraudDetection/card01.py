# /usr/bin/python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set_style('whitegrid')
import missingno as msno

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve
from sklearn.metrics import  recall_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda  x: '%.4f' % x)

from imblearn.over_sampling import SMOTE
import itertools


# -----------------数据预处理-------------------------------------------------
data_cr = pd.read_csv('creditcard.csv', encoding='latin-1')
print data_cr.head() # 查看表格默认前五行
print data_cr.shape  # 查看数据集的大小
print data_cr.info()  # 查看数据的基本信息

print data_cr.describe().T # 查看数据的基本统计信息
print msno.matrix(data_cr) # 查看数据缺失值情况

# -----------------END-------------------------------------------------------


# ----------------特征工程-----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.countplot(x='Class', data=data_cr, ax=axs[0])
axs[0].set_title("Frequency of each Class")
data_cr['Class'].value_counts().plot(x=None,y=None,kind='pie',ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Class")
plt.show()

print data_cr.groupby('Class').size()   # 查看0和1对应的数据集的大小
# ------------------END--------------------------------------------------------

# ----------------特征衍生-------------------------------------------------------
# 特征Time的单为秒，我们将其转化为以小时为单位对应每天的时间。
data_cr['Hour']=data_cr["Time"].apply(lambda  x : divmod(x, 3600)[0]) # 单位转化
# -----------------END----------------------------------------------------------




# -----------------特征选择=-------------------------------------------------------
# 查看信用卡正常用户与被盗用户之间的区别
Xfraud = data_cr.loc[data_cr["Class"] == 1] # update Xfraud & XnonFraud with cleaned data
XnonFraud = data_cr.loc[data_cr["Class"] == 0]

correlationNonFraud = XnonFraud.loc[:, data_cr.columns != 'Class'].corr()
mask = np.zeros_like(correlationNonFraud)
indices = np.triu_indices_from(correlationNonFraud)
mask[indices] = True

grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14, 9))
cmap = sns.diverging_palette(220,8,as_cmap=True)
ax1=sns.heatmap(correlationNonFraud, ax=ax1, vmin=-1, vmax=1, cmap=cmap, square=False, linewidths=0.5, mask=mask, cbar=False)
ax1.set_xticklabels(ax1.get_xticklabels(), size=16)
ax1.set_yticklabels(ax1.get_yticklabels(), size=16)
ax1.set_title('Normal', size=20)

correlationFraud = Xfraud.loc[:, data_cr.columns != 'Class'].corr()
ax2 = sns.heatmap(correlationFraud, vmin=-1, vmax=1, cmap=cmap, ax=ax2, square=False, linewidths=0.5, \
                  mask=mask, yticklabels=False, cbar_ax=cbar_ax, cbar_kws={'orientation': 'vertical', \
                                 'ticks': [-1, -0.5, 0, 0.5, 1]})
ax2.set_xticklabels(ax2.get_xticklabels(), size=16)
ax2.set_title('Fraud', size=20)

cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=14)


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 4))

bins=30

ax1.hist(data_cr["Amount"][data_cr["Class"]==1], bins=bins)
ax1.set_title('Fraud')

ax2.hist(data_cr["Amount"][data_cr["Class"]==0], bins=bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transctions')
plt.yscale('log')
plt.show()

# 以上：信用卡被盗刷发生的金额与信用卡正常用户发生的金额相比呈现散而小的特点，
# 这说明信用卡盗刷者为了不引起信用卡卡主的注意，更偏向选择小金额消费。


sns.factorplot(x="Hour", data=data_cr, kind="count", palette="ocean", size=6, aspect=3)
f, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(16,6))
ax1.scatter(data_cr["Hour"][data_cr["Class"] == 1], data_cr["Amount"][data_cr["Class"] == 1])
ax1.set_title('Fraud')

ax2.scatter(data_cr["Hour"][data_cr["Class"] == 0], data_cr["Amount"][data_cr["Class"] == 0])
ax2.set_title('Normal')

plt.xlabel('Time(in Hours')
plt.ylabel('Amount')
plt.show()

print ("Fraud Stats Summary")
print (data_cr["Amount"][data_cr["Class"] == 1].describe())
print ()
print ("Normal Stats Summary")
print (data_cr["Amount"][data_cr["Class"] == 0].describe())


# 以上、说明信用卡盗刷者为了不引起信用卡卡主注意，更喜欢选择信用卡卡主睡觉时间和消费频率较高的时间点作案；
# 同时，信用卡发生被盗刷的最大值也就只有2,125.87美元。


# Select only the anonymized features
v_feat = data_cr.ix[:, 1:29].columns
plt.figure(figsize=(160, 112))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(data_cr[v_feat]):
    ax = plt.subplot(gs[i])
    sns.distplot(data_cr[cn][data_cr["Class"] == 1], bins=50)
    sns.distplot(data_cr[cn][data_cr["Class"] == 0], bins=100)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
    # plt.show()   # 画出29个图



droplist = ['V8','V13','V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
data_new = data_cr.drop(droplist, axis = 1)
print data_new.shape # 查看数据的维度
# ---------------------END-------------------------------------------------------------






