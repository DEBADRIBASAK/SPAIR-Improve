

import torch
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli, utils
from module import NumericalRelaxedBernoulli
from common import *
import os

box1 = torch.zeros(3, 21, 21)
box1[0, :2, :] = 1
box1[0, -2:, :] = 1
box1[0, :, :2] = 1
box1[0, :, -2:] = 1
box1 = box1.view(1, 3, 21, 21)

box2 = torch.zeros(3, 21, 21)
box2[1, :2, :] = 1
box2[1, -2:, :] = 1
box2[1, :, :2] = 1
box2[1, :, -2:] = 1
box2 = box2.view(1, 3, 21, 21)


box3 = torch.zeros(3, 21, 21)
box3[2, :2, :] = 1
box3[2, -2:, :] = 1
box3[2, :, :2] = 1
box3[2, :, -2:] = 1
box3 = box3.view(1, 3, 21, 21)

box4 = torch.zeros(3, 21, 21)
box4[[0,1], :2, :] = 1
box4[[0,1], -2:, :] = 1
box4[[0,1], :, :2] = 1
box4[[0,1], :, -2:] = 1
box4 = box4.view(1, 3, 21, 21)

box5 = torch.zeros(3, 21, 21)
box5[[0,2], :2, :] = 1
box5[[0,2], -2:, :] = 1
box5[[0,2], :, :2] = 1
box5[[0,2], :, -2:] = 1
box5 = box5.view(1, 3, 21, 21)

box6 = torch.zeros(3, 21, 21)
box6[[1,2], :2, :] = 1
box6[[1,2], -2:, :] = 1
box6[[1,2], :, :2] = 1
box6[[1,2], :, -2:] = 1
box6 = box6.view(1, 3, 21, 21)

box7 = torch.zeros(3, 21, 21)
box7[:, :2, :] = 1
box7[:, -2:, :] = 1
box7[:, :, :2] = 1
box7[:, :, -2:] = 1
box7 = box7.view(1, 3, 21, 21)

box8 = torch.zeros(3, 21, 21)
box8[0, :2, :] = 1
box8[0, -2:, :] = 1
box8[0, :, :2] = 1
box8[0, :, -2:] = 1
box8 = box8.view(1, 3, 21, 21)

box9 = torch.zeros(3, 21, 21)
box9[0, :2, :] = 0.5
box9[0, -2:, :] = 0.5
box9[0, :, :2] = 0.5
box9[0, :, -2:] = 0.5
box9 = box9.view(1, 3, 21, 21)

box10 = torch.zeros(3, 21, 21)
box10[1, :2, :] = 0.5
box10[1, -2:, :] = 0.5
box10[1, :, :2] = 0.5
box10[1, :, -2:] = 0.5
box10 = box10.view(1, 3, 21, 21)


boxes = torch.zeros(N_TOTAL,3,21,21)
N_OBJECTS1 = 16
boxes[:N_OBJECTS1,:,:,:] = box1
boxes[N_OBJECTS1:2*N_OBJECTS1,:,:,:] = box2
boxes[2*N_OBJECTS1:3*N_OBJECTS1,:,:,:] = box3
boxes[3*N_OBJECTS1:4*N_OBJECTS1,:,:,:] = box4
boxes[4*N_OBJECTS1:5*N_OBJECTS1,:,:,:] = box5
boxes[5*N_OBJECTS1:6*N_OBJECTS1,:,:,:] = box6
boxes[6*N_OBJECTS1:7*N_OBJECTS1,:,:,:] = box7
boxes[7*N_OBJECTS1:8*N_OBJECTS1,:,:,:] = box8
boxes[8*N_OBJECTS1:9*N_OBJECTS1,:,:,:] = box9
boxes[9*N_OBJECTS1:10*N_OBJECTS1,:,:,:] = box10

boxes1 = torch.cat((box1,box2,box3,box4,box5,box6,box7,box8,box9,box10,),dim=0)


def visualize(x, z_pres, z_where_scale, z_where_shift, boxes=boxes):
    """
        x: (bs, 3, img_h, img_w)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    #bs = z_pres.size(0)
    num_obj = N_TOTAL
    z_pres = z_pres.view(-1, 1, 1, 1) # (bs*N_TOTAL)
    bs = z_pres.size(0)//boxes.shape[0]
    # z_scale = z_where[:, :, :2].view(-1, 2)
    # z_shift = z_where[:, :, 2:].view(-1, 2)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = spatial_transform(((z_pres.view(-1,1,1,1)>=0.5).float() * torch.cat(bs*(boxes,),dim=0)).view(-1,3,21,21),
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([bs * num_obj, 3, img_h, img_w]),
                             inverse=True)
    bbox = bbox.view(bs,N_OBJECTS,-1,3,img_h,img_w).sum(2)
    #bbox = 1.0-bbox
    bbox = (bbox + torch.stack((x,)*N_OBJECTS,dim=1).view(-1, N_CHANNELS, img_h, img_w)).clamp(0.,1.)*255.0
    return bbox

def visualize1(x, z_pres, z_where_scale, z_where_shift,pred, boxes=boxes):
    """
        x: (bs, 3, img_h, img_w)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    bs = z_pres.size(0)
    num_obj = 4*4
    z_pres = z_pres.view(-1, 1, 1, 1) # (bs*N_TOTAL)
    # z_scale = z_where[:, :, :2].view(-1, 2)
    # z_shift = z_where[:, :, 2:].view(-1, 2)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = spatial_transform(((z_pres.view(-1,1,1,1)>=0.5).float() * boxes1[pred.view(-1)]).view(-1,3,21,21),
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([bs * num_obj, 3, img_h, img_w]),
                             inverse=True)
    #bbox = bbox.view(bs,N_OBJECTS,-1,3,img_h,img_w).sum(2)
    #bbox = 1.0-bbox
    bbox = (bbox + torch.stack((x,)*num_obj,dim=1).view(-1, N_CHANNELS, img_h, img_w)).clamp(0.,1.)*255.0
    return bbox


def print_spair_clevr(global_step, epoch, local_count, count_inter,
                      num_train, total_loss, log_like, z_what_kl_loss, z_where_kl_loss,
                      z_pres_kl_loss, z_depth_kl_loss,classification_loss):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} '.format(global_step, epoch, local_count, num_train),
          '({:3.1f}%)]    '.format(100. * local_count / num_train),
          'total_loss: {:.4f} log_like: {:.4f} '.format(total_loss.item(), log_like.item()),
          'What KL: {:.4f} Where KL: {:.4f} '.format(z_what_kl_loss.item(), z_where_kl_loss.item()),
          'Pres KL: {:.4f} Depth KL: {:.4f} classification_loss: {:.4f}'.format(z_pres_kl_loss.item(), z_depth_kl_loss.item(),classification_loss.item()))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count,
              batch_size, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'batch_size': batch_size,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'num_train': num_train
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=device)
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            print('loading part of model since key check failed')
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in checkpoint['state_dict'].items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            model.load_state_dict(state_dict)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))

        return step, epoch


def linear_annealing(x, step, start_step, end_step, start_value, end_value):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=x.device)
    elif step > end_step:
        x = torch.tensor(end_value, device=x.device)

    return x


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    #print("imgs shape = ",image.shape)
    #print("z where shape = ",z_where.shape)
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-9)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)

def calc_kl_z_pres_bernoulli(z_pres_logits, prior_pres_prob, eps=1e-15):
    z_pres_probs = torch.sigmoid(z_pres_logits).view(-1, 4 * 4)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + \
         (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))

    return kl

def calc_count_acc(z_pres, target):
    # (bs)
    out = (z_pres > .5).float().flatten(start_dim=1).sum(dim=1)

    acc = (out == target.float()).float().mean()

    return acc.item()


def calc_count_more_num(z_pres, target):
    # (bs)
    out = (z_pres > .5).float().flatten(start_dim=1).sum(dim=1)

    more_num = (out > target.float()).float().sum()

    return more_num.item()

