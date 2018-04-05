import sys
import csv 
import numpy as np
from utils import sigmoid

# read model
w, mean, std = np.load('model_log.npy')

test_x = []

n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row != 0:
        test_x.append([float(t) for t in r])
    n_row += 1


# standardize data
test_x = np.transpose(test_x)
normalized_data = []
for i, d in enumerate(test_x):
    normalized_data.append((d-mean[i])/std[i])

test_x = normalized_data
test_x = np.transpose(test_x)

# bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# predict
hypo = [sigmoid(t) for t in np.dot(test_x,w)]
ans = []
for i in range(len(test_x)):
    ans.append([str(i+1)])
    ans[i].append(1 if hypo[i] > 0.5 else 0)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
