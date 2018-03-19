import sys
import csv 
import math
import random
import numpy as np

data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0)) 
    n_row = n_row+1
text.close()

x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# x = np.concatenate((x,x**2), axis=1)
# 增加平方項

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
# 增加bias項    

l_x = len(x[0])

# remove outliers
remove_idx = []
for i in range(len(x)):
    for j in (x[i][82:91] + y[i]):
        if j > 200 or j < 1:
            remove_idx.append(i)
            break

x = np.delete(x, remove_idx, 0)
y = np.delete(y, remove_idx, 0)

# train
w = np.zeros(l_x)         # initial weight vector
lr = 100                        # learning rate
iter = 100000                      # iteration

x_t = x.transpose()
s_gra = np.zeros(l_x)

for i in range(iter):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra/ada
    if (i+1) % 100 == 0: print ('iteration: %d | Cost: %f  ' % ( i+1,cost_a))

# save model
np.save('model.npy',w)