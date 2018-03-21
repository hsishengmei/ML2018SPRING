import sys
import csv 
import math
import random
import numpy as np

# read model
w, mean, std = np.load('model.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append((float(r[i])-mean[n_row%18])/std[n_row%18] )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                n = float(r[i])
            else:
                n = 0
            test_x[n_row//18].append((n-mean[n_row%18])/std[n_row%18] )
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
# 增加bias項  

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    a = a*std[9] + mean[9]
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()