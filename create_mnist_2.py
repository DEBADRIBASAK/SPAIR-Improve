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
parser.add_argument("--pickle_path", type=str, default="./training_mnist_modified.pt")
parser.add_argument("--n_channels",type=int,default=1)

args = parser.parse_args()

# f = open("skeletonized_img.pkl","rb")
# arr = pickle.load(f)

mnist_trainset = datasets.MNIST(root=args.datapath, train=True, download=True, transform=None)

loader = torch.utils.data.DataLoader(mnist_trainset)
arr = loader.dataset.data
print(arr.shape)
num = arr.shape[0]

def isInside(l,p):
	for p1 in l:
		if p[0]>p1[0] and p[0]<p1[0]+28 and p[1]>p1[1] and p[1]<p1[1]+28:
			return True
		if p[0]+27>p1[0] and p[0]+27<p1[0]+28 and p[1]>p1[1] and p[1]<p1[1]+28:
			return True
		if p[0]>p1[0] and p[0]<p1[0]+28 and p[1]+27>p1[1] and p[1]+27<p1[1]+28:
			return True
		if p[0]+27>p1[0] and p[0]+27<p1[0]+28 and p[1]+27>p1[1] and p[1]+27<p1[1]+28:
			return True
	return False

t = []
gt = []
cnt = []

for i in tqdm(range(args.num_samples)):
	points = []
	num_imgs = np.random.randint(3)+1 # 1 to 5 images
	imgs = np.random.randint(num,size=(num_imgs,))
	a = torch.zeros((args.n_channels,args.size,args.size)) 
	b = torch.zeros((args.n_channels,args.size,args.size))
	for ind in imgs:
		x = np.random.randint(99)
		y = np.random.randint(99)
		f = isInside(points,(x,y))
		while f:
			x = np.random.randint(99)
			y = np.random.randint(99)
			f = isInside(points,(x,y))
		points.append((x,y))
		a[:,x:x+28,y:y+28]+=arr[ind,:,:].float()
		b[:,x:x+28,y:y+28]+=torch.ones(arr[ind,:,:].shape)
	t.append(a)
	gt.append(b)
	cnt.append(torch.tensor(num_imgs))
	if(i%10==0):
		print("Iter: ",i)

t = torch.stack(t,dim=0).view(-1,args.n_channels,128,128)
gt = torch.stack(gt,dim=0).view(-1,args.n_channels,128,128)
cnt = torch.stack(cnt,dim=0).view(-1,1,1,1)
ds = TensorDataset(t,gt,cnt)
torch.save(ds,args.pickle_path)
print("Done!")