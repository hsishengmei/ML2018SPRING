import csv
import numpy as np
import random
import sys
from utils import sigmoid, rnd_shuffle

if __name__ == '__main__':
    # parse data
    data = []

    n_row = 0
    text = open(sys.argv[1], 'r') 
    row = csv.reader(text , delimiter=",")

    for r in row:
        if n_row != 0:  data.append([float(k) for k in r])
        n_row = n_row+1

    text.close()
    x = np.array(data)


    # standardize data
    data = np.transpose(data)
    mean = []
    std = []
    for d in data:
        mean.append(np.mean(d))
        std.append(np.std(d))

    normalized_data = []
    for i, d in enumerate(data):
        normalized_data.append((d-mean[i])/std[i])

    data = normalized_data
    x = np.transpose(data)

    y = []
    n_row = 0
    f = open(sys.argv[2], 'r') 
    for _ in range(len(x)):
    # row = csv.reader(text , delimiter=",")
        y.append(int(f.readline()))
    y = np.array(y)
    
    # shuffle
    
    x, y = rnd_shuffle(x, y)

    validate = False
    # take out validation data
    if validate:
        n_fold = 5
        n = random.sample(range(len(x)), int(len(x)/n_fold))
        x_validate = x[n]
        y_validate = y[n]
        x = np.delete(x, n, 0)
        y = np.delete(y, n, 0)

    # train
    data_size = len(x)
    
    dim = len(x[0])
    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    for i in range(data_size):
        if y[i] == 1:
            mu1 += x[i]
            cnt1 += 1
        else:
            mu2 += x[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim, dim))
    sigma2 = np.zeros((dim, dim))
    for i in range(len(x)):
        if y[i] == 1:
            sigma1 += np.dot(np.transpose([x[i] - mu1]), [x[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([x[i] - mu2]), [x[i] - mu2])
    sigma1 /= cnt1
    sigma2 /= cnt2

    shared_sigma = (float(cnt1) / data_size) * sigma1 + (float(cnt2) / data_size) * sigma2
    param = (mu1, mu2, cnt1, cnt2, sigma1, sigma2, shared_sigma)
    if validate:
        sigma_inv = np.linalg.pinv(shared_sigma)
        w = np.dot((mu1-mu2), sigma_inv)
        xx = x_validate.T
        b = (-0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) \
            + (0.5) * np.dot(np.dot([mu2], sigma_inv), mu2) + np.log(float(cnt1)/cnt2)
        a = np.dot(w, xx) + b
        yy = [sigmoid(aa) for aa in a]
        hypo = [(1 if t > 0.5 else 0) for t in yy]

        acc = [(1 if a == b else 0) for a, b in zip(hypo, y_validate)]
        print ('validation | Acc: %f' % ( sum(acc) / len(yy) ))


    np.save('model_gen.npy',(param, mean, std))