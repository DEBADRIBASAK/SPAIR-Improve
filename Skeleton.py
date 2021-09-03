import cv2
import numpy as np
from skimage.filters import threshold_utsu,rank
from skimage.morphology import skeletonize
import scipy.io as sio
import matplotlib.pyplot as plt
import torch


mat = sio.loadmat("../train_32x32.mat")
imgs = mat['X']

num_images = imgs.shape[-1]

train = []

# skeletonize

k = np.ones((32,32))

for i in range(num_images):
	img = cv2.cvtColor(imgs[:,:,:,i],cv2.COLOR_BGR2GRAY)
	thresh = 