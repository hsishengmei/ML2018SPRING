import csv
import numpy as np
from sklearn.model_selection import train_test_split

### 
import tensorflow as tf
tf.set_random_seed(7)
###

from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.initializers import *
from keras.preprocessing.image import ImageDataGenerator

def train(X, Y, model_path = 'trainer/tmp/model_660_2.h5'):
    
    X = X[:,:,:,np.newaxis] / 255
    X, valX, Y, valY = train_test_split(X, Y, test_size=0.2, random_state=5)
    
    input_shape = (48,48,1)  # 48*48 with single channel
    kernel_size = (3,3)
    pool_size = (2,2)
    
    # Build sequential NN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size,
                            padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Conv2D(64, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Conv2D(128, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Flatten())
    
    
    act = advanced_activations.PReLU(init='zero', weights=None)
    model.add(Dense(64))
    model.add(act)
    model.add(Dropout(0.5))
    act2 = advanced_activations.PReLU(init='zero', weights=None)
    model.add(Dense(64))
    model.add(act2)
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath=model_path,
                                   monitor='val_acc',
                                   save_best_only=True,
                                   verbose=2)
    
    # Augmentation on image data
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.25)
                       
    train_datagen.fit(X)
    
    model.fit_generator(train_datagen.flow(X, Y, batch_size=128),
              #steps_per_epoch=X.shape[0]//16+4, 
              epochs=250,
              validation_data=(valX, valY),
              callbacks=[checkpointer],
              verbose=2)
    
    return load_model(model_path)

if __name__ == '__main__':
    X = np.load('tmp/x_train.npy')
    Y = np.load('tmp/y_train.npy')
    train(X, Y, model_path = 'tmp/model_660_2.h5')