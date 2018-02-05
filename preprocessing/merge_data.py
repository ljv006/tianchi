#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

dir = '../input/'
train = pd.read_table(dir + 'train_20171215.txt',engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt',engine='python')

# 因为第一赛季只是预测与时间相关的cnt的数量
# 所以可以对数据以dat和dow进行数据合并
train = train.groupby(['date','day_of_week'],as_index=False).cnt.sum()
plt.plot(train['day_of_week'],train['cnt'],'*')
plt.show()

for i in range(7):
    tmp = train[train['day_of_week']==i+1]
    plt.subplot(7, 1, i+1)
    plt.plot(tmp['date'],tmp['cnt'],'*')
plt.show()