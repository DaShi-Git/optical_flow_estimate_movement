from dataloader import DepthDatasetLoader
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import sys
import os
import json
#from pathlib import Path
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
#import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
#from utilis import *
from net_model import Net
#from torch_cluster import knn_graph
from opticalflow import *


torch.cuda.empty_cache()

dd_object = DepthDatasetLoader(dataset_directory = "data0000/")
# print(dd_object[10].keys())
# print(dd_object[10]['rgb'].shape)
#print(dd_object[192]['filenames'])
#dd_object[10]
print(len(dd_object))

batch_size = 1
n_val = 100
n_train = 400
# train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
train_set, val_set = random_split(dd_object, [n_train, n_val],generator=torch.Generator().manual_seed(0))
# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
#print(train_loader)



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Declare Siamese Network
net = Net()
net.load_state_dict(torch.load('/home/shida/optical_flow_estimation/checkpoint_epoch1.pth'))
net.to(device=device)

# Decalre Loss Function
#criterion = ContrastiveLoss()
criterion = nn.MSELoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=4e-05, weight_decay=0.0005)
#criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.CosineEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')

#net.float()
losses=[]
counter=[]
correctnodes=[]
iteration_number = 0

net.eval()
epoch_loss = 0
# with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
with torch.no_grad():
    for batch in val_loader:

        rgb1 = batch['rgb1_file']
        #print(rgb1)
        rgb2 = batch['rgb2_file']
        flowlist=[]
        for k in range(batch_size):

            flow = gunner(rgb1[k],rgb2[k])
            
            flow = np.transpose(flow,(2,0,1))
            #print(flow.shape)
            flowlist.append(flow)



        #             # x = torch.from_numpy(pc1)
        #             # x=x.type(torch.float)
        #             # edge_index = knn_graph(x, k=6)
        #             #print(edge_index.shape)
        #             #print(edge_index)
        #             #net.float()
        #             #x=x.to(device)
        #             #edge_index=edge_index.to(device)/home/shida/optical_flow_estimation/checkpoint_epoch1.pth
        flowlist = np.array(flowlist)
        flowlist = torch.from_numpy(flowlist)
        #print(flowlist.shape)
        out = net(flowlist.float())
        #print(out)
        #             #print(out)
        #             #seg0islable = seg0islable*255
        #             #print(seg0islable[5438])
        groundlist=batch['t']
        print('predicting',out)
        print('truth',groundlist)
        break