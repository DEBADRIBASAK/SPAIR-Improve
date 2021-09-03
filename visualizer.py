import torch
from tensorboardX import SummaryWriter
from spair import Spair
import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.sp = Spair()
	def forward(self,x,global_step,tau):
		global_step = global_step.item()
		tau = tau.item()
		y, log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth = self.sp(x,global_step,tau)
		return y, log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth