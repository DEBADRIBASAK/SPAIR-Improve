%matplotlib inline
%%writefile z_pres_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset,TensorDataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm as tqdm
import matplotlib.pyplot as plt
import os



class CustomDataset(Dataset):
	def __init__(self,path="./training_dataset_for_background.pt"):
		self.ds = torch.load(path)
		self.transforms = transforms.Compose([transforms.ToTensor()])
	def __getitem__(self,ind):
		datapoint,label = self.ds.__getitem__(ind)
		datapoint = self.transforms(datapoint.view(32,32,1).numpy())
		return (datapoint,label)
	def __len__(self):
		return len(self.ds)

ds = CustomDataset()

model = nn.Sequential(nn.Conv2d(1,4,4,2,1),\
	nn.ReLU(),nn.MaxPool2d(2,0,1),nn.Conv2d(4,8,4,2,1),nn.ReLU(),\
	nn.MaxPool2d(2,0,1),nn.Conv2d(8,16,4,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(16,1),nn.Sigmoid())

for param in model.parameters():
	torch.nn.init.xavier_uniform(param)

if torch.cuda.is_available():
  print('yes')
  model = model.to(torch.device('cuda:0'))

loss = nn.BCELoss(reduction="mean")

train,test = torch.utils.data.random_split(ds,lengths=[int(len(ds)*0.8),int(len(ds)*0.2)])
train_sampler = RandomSampler(train,num_samples=8000,replacement=True)

train_loader = DataLoader(train,batch_size=128,sampler=train_sampler)#,num_workers=2)

_,trial= next(enumerate(train_loader))
print("TRial = ",trial[0].shape)

test_sampler = RandomSampler(test,num_samples=2000,replacement=True)
opt = torch.optim.Adam(model.parameters(),lr=1e-5)

test_loader = DataLoader(test,batch_size=128,sampler=test_sampler)#,num_workers=2)
l1 = []
acc1 = []
l2 = []
PATH = "./Classifier_model"
if not os.path.exists(PATH):
	os.mkdir(PATH)
#PATH = os.path.join(PATH,"model.pt")
for epoch in range(500):
	#print("*")
	model.train()
	avg = 0
	num = 0
	for i,batch in enumerate(train_loader):
		#print("+")
		train_data,train_label = batch
		#print(train_data.shape)
		if torch.cuda.is_available():
			train_data = train_data.cuda()
			train_label = train_label.view(-1,1).cuda()
		predictions = model(train_data)
		# print("train_label = ",train_label.view(-1))
		# print("predictions = ",predictions.view(-1))
		l = loss(predictions,train_label)
		opt.zero_grad()
		l.backward()
		opt.step()
		avg+=l.item()
		num+=1
		if(i%100==0):
			l1.append(l.item())
	print("loss at {}-th epoch: {}".format(epoch,avg/num))
	model.eval()
	avg = 0
	lavg = 0
	num = 0
	for i,batch in enumerate(test_loader):
		test_data,test_label = batch
		if torch.cuda.is_available():
			test_data = test_data.cuda()
			test_label = test_label.view(-1,1).cuda()
		predictions = model(test_data)
		#print("test labels = ",test_label.view(-1))
		#print("test predictions = ",predictions.view(-1))
		predictions1 = (predictions>=0.5).float()
		#print("predictions = ",predictions1.view(-1))
		acc = (predictions1==test_label).float().mean()
		ll = loss(predictions,test_label)#.item()
		avg+=acc.item()
		lavg+=ll.item()
		if(i%100==0):
			acc1.append(acc.item())
			l2.append(ll.item())
			_,ax = plt.subplots((4,4))
			test = test_data[:16,:,:,:]
			testl = test_label[:16,:]
			testp = predictions1[:16,:]
			for i1 in range(0,4,1):
				for i2 in range(0,4,1):
					ax[i1,i2].imshow(test[4*i1+i2,:,:,:].squeeze().numpy(),cmap="gray")
					s = 'Label: {} Prediction: {}'.format(testl[4*i1+i2,:,:,:].squeeze().item(),\
						testp[4*i1+i2,:,:,:].squeeze().item())
					ax[i1,i2].title.set_text(s)
			plt.show()
		num+=1
	print("accuracy at {}-th epoch: {} and loss: {}".format(epoch,avg/num,lavg/num))
	if(epoch%100==0):
		model_PATH = os.path.join(PATH,"model_{}th_epoch.pt".format(epoch))
		torch.save(model.state_dict(),model_PATH)
_,ax = plt.subplots(1,3)
ax[0].plot(l1)
ax[1].plot(acc1)
ax[2].plot(l2)
plt.show()
print("Done!")