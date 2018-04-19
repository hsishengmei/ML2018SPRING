import sys
import numpy as np
from sklearn.model_selection import train_test_split

### 
import tensorflow as tf
tf.set_random_seed(7)
###
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils

import os
from parse import parse_xy
import trainer.train_660 as m1
import trainer.train_660_2 as m2
import trainer.train_665 as m3
import trainer.train_673 as m4
import trainer.train_ens1 as m5
import trainer.train_ens2 as m6
from ens import my_ens

if __name__ == '__main__':
    directory = 'trainer/tmp/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    X, Y = parse_xy(sys.argv[1])
    print('parse done')
    
    Y = np_utils.to_categorical(Y)
    models = []
    models.append(m1.train(X, Y))
    models.append(m2.train(X, Y))
    models.append(m3.train(X, Y))
    models.append(m4.train(X, Y))
    models.append(m5.train(X, Y))
    models.append(m6.train(X, Y))
    print('training done')
    
    model = my_ens(models)
    model.save('model.h5')
    print('model saved')
    