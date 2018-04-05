import sys
import csv 
import numpy as np
from utils import sigmoid

# read model
param, mean, std = np.load('model_gen.npy')

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
#test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# predict
mu1, mu2, cnt1, cnt2, sigma1, sigma2, shared_sigma = param 
sigma_inv = np.linalg.pinv(shared_sigma)
w = np.dot((mu1-mu2), sigma_inv)
xx = test_x.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) \
    + (0.5) * np.dot(np.dot([mu2], sigma_inv), mu2) + np.log(float(cnt1)/cnt2)
a = np.dot(w, xx) + b
yy = [sigmoid(aa) for aa in a]
hypo = [(1 if t > 0.5 else 0) for t in yy]

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
