# coding: utf-8

# In[ ]:


import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from PIL import Image
import PIL
from common import *


class MNIST(Dataset):

    def __init__(self, root='./', phase_train=True):
        self.root = os.path.expanduser(root)
        filename = 'training.pt'
        self.data = torch.load(os.path.join(self.root, filename))
        self.count = 0


    def __getitem__(self, index):
        return (self.data[index], 0)#self.count[index])


    def __len__(self):
        return len(self.data)
