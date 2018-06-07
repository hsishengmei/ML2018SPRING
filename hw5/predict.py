import numpy as np
np.random.seed(7)

import sys, os
import pickle
import csv
from utils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
tf.set_random_seed(7)

from keras.models import load_model
from gensim.parsing.preprocessing import *

def preprocess_docs(docs):
    filters = [lambda x: x.lower(), stem_text, strip_multiple_whitespaces]
    docs = trim_list(docs)
    tmp = [' '.join(preprocess_string(s, filters=filters)) for s in docs]
    docs = tmp.copy()
    del tmp
    return docs

if __name__ == '__main__':
    directory = 'data/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    train = True
    self_train = False
    seq_len = 40
    
    print('load tokenizer...')
    tokenizer_path = 'tokenizer.pickle'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print('load model...')
    model = load_model('model.h5')
    
    print('parse test...')
    docs_test = parse_test(sys.argv[1])

    print('preprocess test...')
    docs_test = preprocess_docs(docs_test)
    
    seq_test = tokenizer.texts_to_sequences(docs_test)
    testX = pad_sequences(seq_test, maxlen = seq_len)
    
    print('predict...')
    hypo = model.predict_classes(testX)
        
    # Output result
    ans = []
    for i in range(len(hypo)):
        ans.append([str(i)])
        ans[i].append(hypo[i][0])
        
    filename = sys.argv[2]
    with open(filename, "w+") as f:
        s = csv.writer(f,delimiter=',',lineterminator='\n')
        s.writerow(["id","label"])
        for i in range(len(ans)):
            s.writerow(ans[i]) 
    