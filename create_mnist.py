import torch
import torchvision
import torchvision.datasets as datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datapath",type=str,default="./data",help="path to mnist dataset")
parser.add_argument("--num_samples",type=int,default=60000,help="number of images in dataset")
parser.add_argument("--size",type=int,default=128,help="size of image in dataset")
parser.add_argument("--num_digits",type=int,default=11,help="maximum number of digits in a single image")

args = parser.parse_args()
mnist_trainset = datasets.MNIST(root=args.datapath, train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root=args.datapath, train=False, download=True, transform=None)



num_samples = args.num_samples
from random import random
training = []
for ind in range(num_samples):
    num_digits = np.random.randint(args.num_digits)
    indices = np.random.randint(60000,size=(num_digits))
    a = torch.zeros(args.size,args.size)
    for i in indices:
        x = np.random.randint(args.size-28)
        y = np.random.randint(args.size-28)
        a[x:x+28,y:y+28]+=dt[i,:,:].float()
    training.append(a)    

t = torch.stack(trainin)