%%writefile demo_Server.py
import json
import torch
from spair_semi_supervised import Spair
from flask import Flask,request
from flask_ngrok import run_with_ngrok
import random
import numpy as np
from torchvision.utils import make_grid

model = Spair(sigma=0.3)
device = torch.device("cuda")
model = torch.nn.DataParallel(model)
model.train()
from utils_further_modified import *


classfier_model = nn.Sequential(nn.Conv2d(1,4,4,2,1),nn.ReLU(),\
            nn.MaxPool2d(2,0,1),nn.Conv2d(4,8,4,2,1),nn.ReLU(),nn.MaxPool2d(2,0,1),\
            nn.Conv2d(8,16,4,2,1),nn.ReLU(),nn.Conv2d(16,32,4,2,1),nn.ReLU(),\
            nn.Dropout(0.2,inplace=True),nn.Flatten(),nn.Linear(32,10),nn.LogSoftmax(dim=1))

state = torch.load("./mnist_model.pt")
classifier_model.load_state_dict(state)

app = Flask(__name__)
run_with_ngrok(app)

load_ckpt(model,None,"./semi_supervised_model_3/ckpt_epoch_10.00.pth",device)

ds = torch.load("./training_mnist_modified.pt")
l = len(ds)


@app.route('/',methods=['GET','POST'])
def index():
	if request.method=='POST':
		print("A request arrived!")
		index = np.random.choice(l)
		img,gt,cnt = tuple(ds.__getitem__(index))
		img = img.unsqueeze(0)
		recon_x, log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, classifier_loss, log = model(img.to(device), 1000, 0.5)
		z_where = log['z_where']
		z_pres = log['z_pres']
		theta = torch.zeros(16,2,3)
		theta[:,0,0] = z_where[:,0]
		theta[:,1,1] = z_where[:,1]
		theta[:,0,-1] = z_where[:,2]
		theta[:,1,-1] = z_where[:,3]

		grid = torch.nn.functional.affine_grid(theta,torch.Size((16,1,64,64)))
		gl = torch.nn.functional.grid_sample(torch.cat((batch[0][1].unsqueeze(0),)*16),grid)

		targets = classifier_model((gl-0.5)/0.5)

		bs = 1
		log = {
		'z_where': log['z_where'],
		'z_what': log['z_what'].view(-1, 4 * 4, z_what_dim),
		'z_where_scale': log['z_where'].view(-1, 4 * 4, z_where_scale_dim + z_where_shift_dim)[:, :, :z_where_scale_dim],
		'z_where_shift': log['z_where'].view(-1, 4 * 4, z_where_scale_dim + z_where_shift_dim)[:, :, z_where_scale_dim:],
		'z_pres': log['z_pres'].permute(0, 2, 3, 1),
		'pred': log['pred'],
		'acc': log['acc']
		}
		bbox = visualize1(img[:1].cpu(), log['z_pres'][:1].cpu().detach(),\
                    log['z_where_scale'][:1].cpu().detach(),\
                    log['z_where_shift'][:1].cpu().detach(),torch.argmax(targets[:16].cpu().detach(),dim=1))
		grid_image = make_grid(bbox[0:16], 4, normalize=True, pad_value=1)
		return json.dumps({
			'gt': grid_image.tolist(),
			'recon': recon_x.tolist(),
			'img': img.tolist(),
			})
	return 'Welcome!!'

if __name__=="__main__":
	app.run()