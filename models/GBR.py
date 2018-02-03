#coding=utf-8
from sklearn import cross_validation
from sklearn import svm
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
import numpy as np

train = pd.read_table('../input/train_20171215.txt',engine='python')
train.describe()

actions1 = train.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'],axis = 1).values

# 切分数据（训练集和测试集）
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=100, test_size=0.2,random_state=0)

print "GradientBoostingRegressor"
gbdt = GradientBoostingRegressor()
for train, test in cv:
    gbdt.fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1,df_train_target[test]))
    print '......'

test_A = pd.read_table('../input/test_A_20171225.txt', engine='python')
test_A.describe()
df_test_data = test_A.values
test_result = gbdt.predict(df_test_data)
# test_A['result'] = test_result
test_A['result'] = test_result.astype('int32')
test_A[['date','result']].to_csv('../data/result.txt',index=False,header=False,sep='\t')