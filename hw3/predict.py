import numpy as np
import csv
import sys
from keras.models import Model, load_model
from parse import parse_test
import os

if __name__ == '__main__':
    directory = 'trainer/tmp/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    testX = parse_test(sys.argv[1])
    testX = testX[:,:,:,np.newaxis] / 255
    modelEns = load_model('model_'+sys.argv[3]+'.h5')
    result = modelEns.predict(testX)
    hypo = [np.argmax(r) for r in result]

    ans = []
    for i in range(len(testX)):
        ans.append([str(i)])
        ans[i].append(hypo[i])
        
    filename = sys.argv[2]
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()