import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, average

# code reference: https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f

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

if __name__ == '__main__':
    m1 = load_model('model1.h5')
    m2 = load_model('model2.h5')
    m3 = load_model('model3.h5')
    m4 = load_model('model4.h5')
    m5 = load_model('model5.h5')
    m6 = load_model('model6.h5')
    models = [m1, m2, m3, m4, m5, m6]
    
    model = my_ens(models)
    model.save('ens_model.h5')