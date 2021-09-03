import json
import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import torch
from utils_modified import *

URI = "http://3f239e11952f.ngrok.io"
st.title("SPAIR Visulaizer")

if st.button('Get Random Prediction'):
    response = requests.post(URI,data={})
    response = json.loads(response.text)
    #preds = response.get('preds')
    # img = response.get('img')
    # img = torch.tensor(img).squeeze()
    # img = torch.reshape(img,(128,128))
    # img = img.clamp(0.0,1.0)
    # st.sidebar.markdown("### Input Image")
    # st.sidebar.image(img.cpu().detach().numpy(),width=200)

    gt = response.get('pred')
    gt = torch.tensor(gt).squeeze().permute(1,2,0)
    gt = gt.clamp(0.0,1.0)
    st.image(gt.cpu().detach().squeeze().numpy(),width=500)

    gl = response.get('glimpse')
    gl = torch.tensor(gl)

    target = response.get('z_pres')
    target = torch.tensor(target)

    plt.figure(figsize=(32,4))
    row = 2
    col = 8
    for i in range(16):
            plt.subplot(row,col,i+1)
            plt.imshow(gl[i,:,:,:].squeeze().detach().cpu().numpy().astype('float32'),cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("{:.2f}".format(target[i].item()),fontsize=40)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    #plt.tight_layout()
    st.pyplot()
        