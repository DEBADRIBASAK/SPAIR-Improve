%%writefile demo_Server_2.py
import json
import torch
import torch.nn as nn
from utils_modified import *
from spair_with_z_pres_guidance import Spair
device = torch.device("cuda:0")
trained_model_path = "./Classifier_model_2/model_499th_epoch.pt"
model = nn.Sequential(nn.Conv2d(1,4,4,2,1),\
        nn.ReLU(),nn.MaxPool2d(2,0,1),nn.Conv2d(4,8,4,2,1),nn.ReLU(),\
        nn.MaxPool2d(2,0,1),nn.Conv2d(8,16,4,2,1),nn.ReLU(),\
        nn.Flatten(),nn.Linear(16,1),nn.Sigmoid())
spair_model = Spair("./Classifier_model_2/model_499th_epoch.pt")
spair_model = nn.DataParallel(spair_model)
load_ckpt(spair_model,None,"Ijjat_model_2/ckpt_epoch_4.00.pth",device=device)
spair_model.eval()
from flask import Flask,request
from flask_ngrok import run_with_ngrok
import random
import numpy as np
from torchvision.utils import make_grid



app = Flask(__name__)
run_with_ngrok(app)


ds = torch.load("./aspect_reserved.pt")
l = len(ds)


@app.route('/',methods=['GET','POST'])
def index():
	if request.method=='POST':
		print("A request arrived!")
		index = np.random.choice(l)
		img,gt,cnt = tuple(ds.__getitem__(index))
		img = img.unsqueeze(0)
		recon_x, log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg_what, classifier_loss, log = \
		spair_model(img, 1000,0.5)
		glimpses = spatial_transform(torch.stack(4*4*(img.unsqueeze(0),),dim=1).view(-1,1,128,128).float(),log['z_where'],(4*4*1,1,32,32),inverse=False)
		gl = (glimpses-0.5)/0.5
		pred = model(gl)
		log1 = {'z_where_scale': log['z_where'].view(-1, 4 * 4, 4)[:, :, :2], 'z_where_shift': log['z_where'].view(-1, 4 * 4, 4)[:, :, 2:],}
		bbox = visualize(img.unsqueeze(0).cpu(), pred.cpu().detach().view(1,4,4),\
			log1['z_where_scale'].cpu().detach(),\
			log1['z_where_shift'].cpu().detach())
		grid_image = make_grid(bbox[0:16], 4, normalize=True, pad_value=1)
		return json.dumps({
			'img': img.tolist(),
			'pred': grid_image.tolist(),
			'z_pres': pred.tolist(),
			'glimpse': glimpses.tolist()
			})
	return 'Welcome!!'

if __name__=="__main__":
	app.run()