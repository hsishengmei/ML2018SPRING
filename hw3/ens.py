import numpy as np
import csv
from keras.models import Model, load_model
from keras.layers import Input, average

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')  
   
    return modelEns

def my_ens(model_list):
    model_input = Input(shape=model_list[0].input_shape[1:]) # c*h*w
    modelEns = ensembleModels(model_list, model_input)
    return modelEns
