import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import cluster
import sys

if __name__ == '__main__':
    # parse testcase
    test_case = []
    with open(sys.argv[2], 'r') as text: 
        n_row = 0
        row = csv.reader(text , delimiter=",")
        for r in row:
            if n_row != 0: 
                test_case.append([int(r[1]), int(r[2])])
            n_row = n_row+1
    
    # parse image and do reduction
    X = np.load(sys.argv[1]) / 255
    X_PCA = PCA(n_components=200, copy=False, whiten=True, svd_solver='randomized', random_state=7).fit_transform(X)
    
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(X_PCA)
    
    label = k_means.labels_
    
    ans = []
    for i in test_case:
        if label[i[0]] == label[i[1]]:
            ans.append(1)
        else:
            ans.append(0)
    
    with open(sys.argv[3], "w+") as f:
        s = csv.writer(f,delimiter=',',lineterminator='\n')
        s.writerow(["ID","Ans"])
        for i in range(len(ans)):
            s.writerow([i,ans[i]])
    
    
    
    
    
            