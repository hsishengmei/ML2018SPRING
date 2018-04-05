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

    # bias
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


    n_features = len(x[0])
    w_sum = np.zeros(n_features)
    for _ in range(1):
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
        lr = 0.2
        iter = 900
        lamb = 0.02
        w = np.zeros(n_features)         # initial weight vector
        x_t = x.transpose()
        s_gra = np.zeros(n_features)

        for i in range(iter):
            hypo = np.array([sigmoid(t) for t in np.dot(x,w)])
            loss = hypo - y
            cost = np.sum(loss**2) / len(x)
            cost_a = np.sqrt(cost)
            gra = np.dot(x_t,loss) + sum([k*k for k in w]) * lamb
            s_gra += gra**2
            ada = np.sqrt(s_gra)
            w = w - lr * gra/ada
            if (i+1) % 100 == 0: print ('iteration: %d | Cost: %f  ' % ( i+1,cost_a))


        if validate:
            ans = [sigmoid(t) for t in np.dot(x_validate, w)]
            hypo = [(1 if t > 0.5 else 0) for t in ans]
            acc = [(1 if a == b else 0) for a, b in zip(hypo, y_validate)]
            print ('validation | Acc: %f' % ( sum(acc) / len(hypo) ))
            
        '''
        ans = [sigmoid(t) for t in np.dot(x, w)]
        hypo = [(1 if t > 0.5 else 0) for t in ans]
        acc = [(1 if a == b else 0) for a, b in zip(hypo, y)]
        print ('validation | Acc: %f' % ( sum(acc) / len(hypo) ))
        
        remove_idx = []
        if _ == 0:
            for j in range(len(acc)):
                if not acc[j]:
                    remove_idx.append(j)
            
            print(x.shape)
            x = np.delete(x, remove_idx, 0)
            y = np.delete(y, remove_idx, 0)
            print(x.shape)
        '''

    np.save('model_log.npy',(w, mean, std))