import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import argparse
from torch.utils.data import DataLoader, Dataset, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--datapath",type=str,default="./data",help="path to mnist dataset")
parser.add_argument("--num_samples",type=int,default=10000,help="number of images in dataset")
parser.add_argument("--size",type=int,default=128,help="size of image in dataset")
parser.add_argument("--pickle_path", type=str, default="./training1.pt")

#parser.add_argument("--num_digits",type=int,default=11,help="maximum number of digits in a single image")

args = parser.parse_args()
if(not os.path.isfile(args.pickle_path)):
	mnist_trainset = datasets.MNIST(root=args.datapath, train=True, download=True, transform=None)
	#mnist_testset = datasets.MNIST(root=args.datapath, train=False, download=True, transform=None)

	num_samples = args.num_samples
	training = []
	num = []

	dl = DataLoader(mnist_trainset)
	dt = dl.dataset.data

	for ind in range(num_samples):
	    num_digits = 5 #np.random.randint(11)
	    indices = np.random.randint(60000, size=(num_digits))
	    a = torch.zeros(3, 128, 128)
	    for i in indices:
	        x = np.random.randint(70)
	        y = np.random.randint(70)
	        a[:, x:x+28, y:y+28] += dt[i,:,:].float()

	    training.append(a)
	    num.append(torch.tensor([num_digits]))

	training = torch.stack(training, dim=0).squeeze().view(-1, 3, 128, 128)
	num = torch.stack(num, dim=0).squeeze().view(-1, 1, 1, 1)
	ds = torch.utils.data.TensorDataset(training, num)
	#training = torch.stack([training.float(),num.float()],dim=1)
	torch.save(ds, args.pickle_path)
