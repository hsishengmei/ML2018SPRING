import csv
import numpy as np

def parse_xy(path):
    # parse data
    X = []
    Y = []
    
    with open(path, 'r') as text: 
        n_row = 0
        row = csv.reader(text , delimiter=",")
        for r in row:
            if n_row != 0: 
                Y.append(r[0])
                data = r[1].split(' ')
                data = np.array(list(map(int, data)))
                data = data.reshape(48,48)
                X.append(data)
            n_row = n_row+1
        X = np.array(X)
        Y = np.array(Y)
        
    np.save('trainer/tmp/x_train.npy', X)
    np.save('trainer/tmp/y_train.npy', Y)
    return (X, Y)


def parse_test(path):
    test_X = []
    with open(path, 'r') as text: 
        n_row = 0
        row = csv.reader(text , delimiter=",")
        for r in row:
            if n_row != 0: 
                data = r[1].split(' ')
                data = np.array(list(map(int, data)))
                data = data.reshape(48,48)
                test_X.append(data)
            n_row = n_row+1
        test_X = np.array(test_X)
    np.save('trainer/tmp/x_test.npy', test_X)
    return test_X

