import json
import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import torch
from utils_further_modified import *

URI = "http://69aa1ab8a09e.ngrok.io"
st.title("SPAIR Visulaizer")

if st.button('Get Random Prediction'):
    response = requests.post(URI,data={})
    response = json.loads(response.text)
    #preds = response.get('preds')
    img = response.get('img')
    img = torch.tensor(img).squeeze()
    img = torch.reshape(img,(128,128))
    img = img.clamp(0.0,1.0)
    st.sidebar.markdown("### Input Image")
    st.sidebar.image(img.cpu().detach().numpy(),width=200)
    recon = response.get('recon')
    recon = torch.tensor(recon).squeeze()
    recon = torch.reshape(recon,(128,128))
    recon = recon.clamp(0.,1.)
    st.sidebar.markdown("### Reconstructed Image")
    st.sidebar.image(recon.cpu().detach().numpy(),width=200)
    gt = response.get('gt')
    gt = torch.tensor(gt).squeeze().permute(1,2,0)
    gt = gt.clamp(0.0,1.0)
    st.image(gt.cpu().detach().squeeze().numpy(),width=500)

    gl = response.get('gl')
    gl = torch.tensor(gl)

    z_pres = response.get('z_pres')
    z_pres = torch.tensor(z_pres).view(-1)

    target = response.get('target')
    #print(target)
    target = torch.tensor(target).view(-1).tolist()

    #st.text(str(target))

    plt.figure(figsize=(32,4))
    row = 2
    col = 8
    for i in range(16):
        if z_pres[i]==True:
            plt.subplot(row,col,i+1)
            plt.imshow(gl[i,:,:,:].squeeze().detach().cpu().numpy().astype('float32'),cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(str(target[i]),fontsize=40)
           
        else:
            plt.subplot(row,col,i+1)
            plt.imshow(np.ones((64,64)).astype('float32'),cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("Nothing!",fontsize=40)

    plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.tight_layout()
    st.pyplot()
        