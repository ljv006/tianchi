#coding=utf-8
from sklearn import cross_validation
from sklearn import svm
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from series_to_supervised import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
train = pd.read_table('../input/train_20171215.txt',engine='python')
train.describe()
time_cnt = list(train['cnt'].values)
# nin 前看 nout后看 这个题目需要前看
time2sup = series_to_supervised(data=time_cnt,n_in=276,dropnan=True)
print(time2sup.shape)
x_train = time2sup[time2sup.index<755]
x_test = time2sup[time2sup.index>755]
# 这个方式其实是最简单的，后面还可以很多改善，比如滚动预测一类
print(x_train.shape)
print(x_test.shape)

y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')

# actions1 = train.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})
#
df_train_target = time2sup['var1(t)'].values
df_train_data = time2sup.drop(['var1(t)'],axis = 1).values
#
# 切分数据（训练集和测试集）
# cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=1, test_size=0.2,random_state=0)

print "GradientBoostingRegressor"
gbdt = GradientBoostingRegressor()
# for train, test in cv:
#     gbdt.fit(df_train_data[train], df_train_target[train])
#     result1 = gbdt.predict(df_train_data[test])
#     print(mean_squared_error(result1,df_train_target[test]))
#     print '......'
# 损失函数mse
gbdt.fit(x_train.values,y_train)
print(gbdt.predict(x_test.values))
print len(gbdt.predict(df_train_data))
from sklearn.metrics import mean_squared_error
line1 = plt.plot(range(len(x_test)),gbdt.predict(x_test.values),label=u'predict')
line2 = plt.plot(range(len(y_test)),y_test.values,label=u'true')
plt.legend()
plt.show()
# test_A = pd.read_table('../input/test_A_20171225.txt', engine='python')
# test_A.describe()
# df_test_data = test_A.values
# test_result = gbdt.predict(df_test_data)
# # test_A['result'] = test_result
# test_A['result'] = test_result.astype('int32')
# test_A[['date','result']].to_csv('../data/result.txt',index=False,header=False,sep='\t')