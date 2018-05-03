from skimage import io
import numpy as np
import os
import sys

def to_img(img):
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 255).astype(np.uint8)
    img = img.reshape(600,600,3)
    return img

# load image
img_list = []
dir_path = sys.argv[1]
files = os.listdir(dir_path)

for f in files:
    if f[-4:] == '.jpg':
        fname = dir_path + '/' + f
        img = io.imread(fname).flatten()
        img_list.append(img)

img_mean = np.mean(img_list, axis=0)

# run SVD
U, s, V = np.linalg.svd((img_list - img_mean).T, full_matrices=False)
    
# reconstruct image
target_name = dir_path + '/' + sys.argv[2]
target_img = io.imread(fname).flatten()

n_eig = 4
weights = np.dot(target_img - img_mean, U[:,:n_eig])
rec = img_mean
for i in range(n_eig):
    rec = rec + U.T[i] * weights[i]
rec = to_img(rec)
io.imsave('reconstruction.jpg', rec)

    
