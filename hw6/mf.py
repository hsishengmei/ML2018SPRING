import numpy as np 
import sys
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, dot, Embedding, add, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.constraints import non_neg
tf.set_random_seed(7)
np.random.seed(7)
model_name = 'MF.h5'

def parse_train(filename = 'data/train.csv'):
    users = []
    movies = []
    ratings = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            ft = line.strip('\n').split(',')
            ft = list(map(int, ft))
            users.append(ft[1])
            movies.append(ft[2])
            ratings.append(ft[3])
    return np.array(users), np.array(movies), np.array(ratings)

def parse_test(filename = 'data/test.csv'):
    users = []
    movies = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            ft = line.strip('\n').split(',')
            ft = list(map(int, ft))
            users.append(ft[1])
            movies.append(ft[2])
    return np.array(users), np.array(movies)

train = False
if train:
    users, movies, ratings = parse_train()
    users_train, users_val, movies_train, movies_val, ratings_train, ratings_val \
        = train_test_split(users, movies, ratings, test_size=0.1)
    
    n_users = 6040
    n_movies = 3952
    d = 128
    
    user_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=n_users+1, output_dim=d, embeddings_initializer='orthogonal')(user_input)
    user_embedding = Flatten()(user_embedding)
    
    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(input_dim=n_movies+1, output_dim=d, embeddings_initializer='orthogonal')(movie_input)
    movie_embedding = Flatten()(movie_embedding)  
    
    user_bias = Embedding(input_dim=n_users+1, output_dim=1, embeddings_initializer='orthogonal', embeddings_constraint=non_neg())(user_input)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(input_dim=n_movies+1, output_dim=1, embeddings_initializer='orthogonal', embeddings_constraint=non_neg())(movie_input)
    movie_bias = Flatten()(movie_bias)
    
    preference = dot(inputs=[user_embedding, movie_embedding], axes=1)
    preference = add(inputs=[preference, user_bias, movie_bias])
    preference = Dense(1, bias_initializer='ones', activation='linear')(preference)
    
    model = Model(inputs=[user_input, movie_input], outputs=preference)
    model.compile(loss='mse', optimizer='rmsprop')
    
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath=model_name,
                                   monitor='val_loss',save_best_only=True,
                                   verbose=1)
    
    model.fit([users_train, movies_train], ratings_train,
        batch_size=256, epochs=20,
        validation_data=([users_val, movies_val], ratings_val),
        callbacks=[checkpointer],
        verbose=1)

users_test, movies_test = parse_test(sys.argv[1])
model = load_model(model_name)
hypo = model.predict([users_test, movies_test])

with open(sys.argv[2], 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, res in enumerate(hypo):
        if res[0] > 5: res[0] = 5
        if res[0] < 1: res[0] = 1
        f.write(str(i+1)+','+str(res[0])+'\n')