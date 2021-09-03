import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, Dataset, TensorDataset
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--datapath",type=str,default="./data",help="path to mnist dataset")
parser.add_argument("--num_samples",type=int,default=10,help="number of images in dataset")
parser.add_argument("--size",type=int,default=128,help="size of image in dataset")
parser.add_argument("--pickle_path", type=str, default="./training_skl.pt")
parser.add_argument("--n_channels",type=int,default=1)

args = parser.parse_args()

f = open("skeletonized_img.pkl","rb")
arr = pickle.load(f)

num = arr.shape[0]

def isInside(l,p):
	for p1 in l:
		if p[0]>p1[0] and p[0]<p1[0]+32 and p[1]>p1[1] and p[1]<p1[1]+32:
			return True
		if p[0]+31>p1[0] and p[0]+31<p1[0]+32 and p[1]>p1[1] and p[1]<p1[1]+32:
			return True
		if p[0]>p1[0] and p[0]<p1[0]+32 and p[1]+31>p1[1] and p[1]+31<p1[1]+32:
			return True
		if p[0]+31>p1[0] and p[0]+31<p1[0]+32 and p[1]+31>p1[1] and p[1]+31<p1[1]+32:
			return True
	return False

t = []
gt = []
cnt = []

for i in tqdm(range(args.num_samples)):
	points = []
	num_imgs = np.random.randint(5)+1 # 1 to 5 images
	imgs = np.random.randint(num,size=(num_imgs,))
	a = np.zeros((args.n_channels,args.size,args.size)) 
	b = np.zeros((args.n_channels,args.size,args.size))
	for ind in imgs:
		x = np.random.randint(90)
		y = np.random.randint(90)
		f = isInside(points,(x,y))
		while f:
			x = np.random.randint(90)
			y = np.random.randint(90)
			f = isInside(points,(x,y))
		points.append((x,y))
		a[:,x:x+32,y:y+32]+=arr[ind,:,:]
		b[:,x:x+32,y:y+32]+=np.ones(arr[ind,:,:].shape)
	t.append(a)
	gt.append(b)
	cnt.append(num_imgs)
	if(i%10==0):
		print("Iter: ",i)

t = torch.tensor(t,dtype=torch.float32)
gt = torch.tensor(gt,dtype=torch.float32)
cnt = torch.tensor(cnt,dtype=torch.int8)
ds = TensorDataset(t,gt,cnt)
torch.save(ds,args.pickle_path)