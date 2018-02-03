#coding=utf-8
from sklearn import cross_validation
from sklearn import svm
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np

def print_error(error):
    print "total_cost: " + str(error)
    print "-" * 100

train = pd.read_table('../input/train_20171215.txt',engine='python')
train.describe()

actions1 = train.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'],axis = 1).values

# 切分数据（训练集和测试集）
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=20, test_size=0.2,random_state=0)

print "BayesianRidge"
counter = 1.0
total_cost = 0
for train, test in cv:
    reg = linear_model.BayesianRidge().fit(df_train_data[train], df_train_target[train])
    result1 = reg.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    total_cost += mean_squared_error(result1,df_train_target[test])
    counter += 1.0
    print '......'
print_error(total_cost / counter)

counter = 1.0
total_cost = 0
print "Lasso"
for train, test in cv:
    reg = linear_model.Lasso(alpha = 0.1).fit(df_train_data[train], df_train_target[train])
    result1 = reg.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    total_cost += mean_squared_error(result1, df_train_target[test])
    counter += 1.0
    print '......'
print_error(total_cost / counter)

counter = 1.0
total_cost = 0
print "GradientBoostingRegressor"

for train, test in cv:
    gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    total_cost += mean_squared_error(result1, df_train_target[test])
    counter += 1.0
    print '......'
print_error(total_cost / counter)

print "MixModel"
counter = 1.0
total_cost = 0
train_temp = pd.read_table('../input/train_20171215.txt',engine='python')
train_temp.describe()
actions = train_temp.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

df_train_target_temp = train_temp.drop(['brand'],axis = 1)['cnt'].values
df_train_data_temp = train_temp.drop(['cnt'],axis = 1).drop(['brand'],axis = 1).values


cv_temp = cross_validation.ShuffleSplit(len(df_train_data_temp), n_iter=10, test_size=0.2,random_state=0)
reg_temp = linear_model.Lasso(alpha=0.1)
gbdt_temp = GradientBoostingRegressor()
gbdt_temp1 = GradientBoostingRegressor()
for train, test in cv_temp:
    reg_temp.fit(df_train_data_temp[train], df_train_target_temp[train])
result0 = reg_temp.predict(df_train_data_temp)
train_temp['new_column1'] = result0
df_train_target_temp = train_temp.drop(['brand'],axis = 1)['cnt'].values
df_train_data_temp = train_temp.drop(['cnt'],axis = 1).drop(['brand'],axis = 1).values

cv_temp = cross_validation.ShuffleSplit(len(df_train_data_temp), n_iter=10, test_size=0.2,random_state=0)
for train, test in cv_temp:
    gbdt_temp.fit(df_train_data_temp[train], df_train_target_temp[train])
result1 = gbdt_temp.predict(df_train_data_temp)
train_temp['new_column2'] = result1
df_train_target_temp = train_temp.drop(['brand'],axis = 1)['cnt'].values
df_train_data_temp = train_temp.drop(['cnt'],axis = 1).drop(['brand'],axis = 1).values

cv_temp = cross_validation.ShuffleSplit(len(df_train_data_temp), n_iter=10, test_size=0.2,random_state=0)
for train, test in cv_temp:
    gbdt_temp1.fit(df_train_data_temp[train], df_train_target_temp[train])

result2 = gbdt_temp1.predict(df_train_data_temp)
train_temp['new_column3'] = result2

actions1 = train_temp.groupby(['date','day_of_week','new_column1','new_column2','new_column3'], as_index=False)['cnt'].agg({'count1':np.sum})

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'],axis = 1).values

cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=10, test_size=0.2,random_state=0)

for train, test in cv:
    gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    total_cost += mean_squared_error(result1, df_train_target[test])
    counter += 1.0
    print '......'
print_error(total_cost / counter)

test_A = pd.read_table('../input/test_A_20171225.txt', engine='python')
test_A.describe()
df_test_data = test_A.values
result0 = reg_temp.predict(df_test_data)
test_A['new_column1'] = result0
df_test_data = test_A.values
result1 = gbdt_temp.predict(df_test_data)
test_A['new_column2'] = result1
df_test_data = test_A.values
result2 = gbdt_temp1.predict(df_test_data)
test_A['new_column3'] = result2
df_test_data = test_A.values
test_result = gbdt.predict(df_test_data)
# test_A['result'] = test_result
test_A['result'] = test_result.astype('int32')
test_A[['date','result']].to_csv('../data/result_temp.txt',index=False,header=False,sep='\t')