import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.morphology import skeletonize
from skimage.filters import rank
import pickle

mat = sio.loadmat('../train_32x32.mat')
imgs = mat['X']
num_imgs = imgs.shape[-1]
sele = np.ones((32,32))
gray_imgs = []

for i in tqdm(range(50000)):
	img = cv2.cvtColor(imgs[:,:,:,i],cv2.COLOR_BGR2GRAY).astype(float)
	thresh = rank.otsu(img/255.0,sele)
	img1 = (img>=thresh).astype(float)
	white_cells = np.sum(img1)
	if((32*32)-white_cells<white_cells):
		img1 = 1.0-img1
	img1 = skeletonize(img1).astype(float)
	img1*=255.0
	gray_imgs.append(img1)

gray_imgs = np.array(gray_imgs)
#with  as f:
f = open('skeletonized_img.pkl','wb')
pickle.dump(np.array(gray_imgs),f)
print('Done!')