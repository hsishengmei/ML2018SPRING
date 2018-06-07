import numpy as np
np.random.seed(7)

import sys
import pickle
import csv
from utils import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
tf.set_random_seed(7)

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras import regularizers

from gensim.parsing.preprocessing import *

def train_model(X, Y, valX, valY, word_count, filepath='model.h5', seq_len=40, n_epoch=7):
    print('train...')
    model = Sequential()
    model.add(Embedding(word_count,50,input_length=seq_len))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='val_acc',save_best_only=True,
                                   verbose=1)
    model.fit(X, Y,
              batch_size=256,
              epochs=n_epoch,
              validation_data=(valX, valY),
              callbacks=[checkpointer])

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
    
    try:
        print('load text...')
        docs1 = np.load('data/docs1.npy')
        docs2 = np.load('data/docs2.npy')
        labels = np.load('data/labels.npy')
    except:
        # parse docs
        print('parse text...')
        docs1, labels = parse_label(filepath=sys.argv[1])
        docs2 = parse_no_label(filepath=sys.argv[2])
    
        # docs preprocessing
        print('preprocessing...')
        docs1 = preprocess_docs(docs1)
        docs2 = preprocess_docs(docs2)
        
        np.save('data/docs1.npy', docs1)
        np.save('data/docs2.npy', docs2)
        np.save('data/labels.npy', labels)
        
    docs = np.concatenate((docs1, docs2))
    
    try:
        print('load tokenizer...')
        tokenizer_path = 'tokenizer.pickle'
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        # tokenizer
        print('train tokenizer...')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(docs)
        with open('tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
            print('Tokenizer saved...')
    
    try:
        X = np.load('data/trainX.npy')
        Y = np.load('data/trainY.npy')
        valX = np.load('data/valX.npy')
        valY = np.load('data/valY.npy')
    except:            
        seq = tokenizer.texts_to_sequences(docs1)
        seq_pad = pad_sequences(seq, maxlen = seq_len)
        
        print('cut validation set...')
        
        val_ratio = 0.2
        val_num = int(len(seq_pad) * val_ratio)
        X = seq_pad[:-val_num]
        Y = labels[:-val_num]
        valX = seq_pad[-val_num:]
        valY = labels[-val_num:]
        
        np.save('data/trainX.npy', X)
        np.save('data/trainY.npy', Y)
        np.save('data/valX.npy', valX)
        np.save('data/valY.npy', valY)
        
    word_count = len(tokenizer.word_docs)
    if train:
        print('train...')
        # train model    
        train_model(X, Y, valX, valY, word_count)
    
    print('load model...')
    model = load_model('model.h5')
    
    '''
    # predict
    try:
        print('load test...')
        seq_test = np.load('data/seq_test.npy')
    except:
        print('parse test...')
        docs_test = parse_test()
        print('preprocess test...')
        docs_test = preprocess_docs(docs_test)
        
        seq_test = tokenizer.texts_to_sequences(docs_test)
        np.save('data/seq_test.npy', seq_test)
    
    testX = pad_sequences(seq_test, maxlen = seq_len)
    
    print('predict...')
    hypo = model.predict_classes(testX)
        
    # Output result
    ans = []
    for i in range(len(hypo)):
        ans.append([str(i)])
        ans[i].append(hypo[i][0])
        
    filename = 'predict_dnn.csv'
    with open(filename, "w+") as f:
        s = csv.writer(f,delimiter=',',lineterminator='\n')
        s.writerow(["id","label"])
        for i in range(len(ans)):
            s.writerow(ans[i]) 
    '''
    
    # self-train
    if self_train:
        print('self-train...')
        try:
            docs2 = np.load('data/docs2_pad.npy')
        except:
            print('preprocess unlabeled data...')
            seq2 = tokenizer.texts_to_sequences(docs2)
            docs2 = pad_sequences(seq2, maxlen = seq_len)
            np.save('data/docs2_pad.npy', docs2)
        
        for k in range(5):
            print('predict unlabeled data... ' + str(i))
            data_semi, label_semi = semi(docs2, model, tokenizer)
            # data_semi = np.load('data/data_semi.npy')
            # label_semi = np.load('data/label_semi.npy')
            data_semi = np.concatenate((X, np.array(data_semi)))
            label_semi = np.concatenate((Y, np.array(label_semi)))
            
            checkpointer = ModelCheckpoint(filepath='model_semi_weights'+str(k)+'.hdf5',
                                           monitor='val_acc',save_best_only=True,
                                           verbose=1)
            model.fit(data_semi, label_semi,
                      batch_size=256,
                      epochs=2,
                      validation_data=(valX, valY),
                      callbacks=[checkpointer])
            print('load semi-model...')
            model.load_weights('model_semi_weights'+str(i)+'.hdf5')
            del data_semi, label_semi
    
            print('predict...')
            hypo = model.predict_classes(testX)
            
            # Output result
            ans = []
            for i in range(len(hypo)):
                ans.append([str(i)])
                ans[i].append(hypo[i][0])
                
            filename = 'predict_semi_'+str(k)+'.csv'
            with open(filename, "w+") as f:
                s = csv.writer(f,delimiter=',',lineterminator='\n')
                s.writerow(["id","label"])
                for i in range(len(ans)):
                    s.writerow(ans[i]) 
    
    
    
    
    
    
    
    