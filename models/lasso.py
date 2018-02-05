#coding=utf-8
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import pandas as pd
import numpy as np

def print_error(error):
    print "total_cost: " + str(error)
    print "-" * 100

train_temp = pd.read_table('../input/train_20171215.txt',engine='python')
train_temp.describe()
actions = train_temp.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

df_train_target_temp = actions['count1'].values
df_train_data_temp = actions.drop(['count1'],axis = 1).values


cv_temp = cross_validation.ShuffleSplit(len(df_train_data_temp), n_iter=10, test_size=0.2,random_state=0)
reg_temp = linear_model.Lasso(alpha=0.2)
for train, test in cv_temp:
    reg_temp.fit(df_train_data_temp[train], df_train_target_temp[train])
actions['new_column1'] = reg_temp.predict(df_train_data_temp)

df_train_target = actions['count1'].values
df_train_data = actions.drop(['count1'],axis = 1).values

# 切分数据（训练集和测试集）
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=100, test_size=0.2,random_state=0)

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

test_A = pd.read_table('../input/test_A_20171225.txt', engine='python')
test_A.describe()
df_test_data = test_A.values
result0 = reg_temp.predict(df_test_data)
test_A_temp = test_A
test_A_temp['new_column1'] = result0
df_test_data = test_A_temp.values
test_result = reg.predict(df_test_data) - 277
# test_A['result'] = test_result
test_A['result'] = test_result.astype('int32')
test_A[['date','result']].to_csv('../data/result.txt',index=False,header=False,sep='\t')