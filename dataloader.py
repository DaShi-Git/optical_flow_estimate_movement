import cv2
import numpy as np
#from torch._C import float32
from torch.utils.data import Dataset, DataLoader
import glob
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io
import torch
import pandas as pd
import random
# def addLayer(img, dirct):
#     leftorright = dirct[17]
#     additionLayer = np.ones((1,512,512))
#     #additionLayer = torch.from_numpy(np.ones((1,512,512)))
#     #img = torch .cat([img, additionLayer],0)
#     #img = np.concatenate((img, additionLayer), axis=0)
#     dis = int(dirct[18:20])/10.0
#     #print(dirct[18:20])
#     if leftorright =='0':
#         img = np.concatenate((img, additionLayer*dis), axis=0)

#     elif leftorright =='l':


#         img = np.concatenate((img, additionLayer*dis), axis=0)


#     elif leftorright =='r':

#         img = np.concatenate((img, additionLayer*(-1*dis)), axis=0)
        


#     return img

def read_rgb(rgb_file):
    rgb = io.imread(rgb_file)
    # plt.imshow(rgb)
    # plt.title(rgb_file)
    # plt.show()
    return rgb

def read_depth(depth_file):
    depth = io.imread(depth_file)
    # Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256)
    depth = depth * (1/ (256 * 256 * 256 - 1))
    # plt.imshow(depth)
    # plt.title(depth_file)
    # plt.show()
    return depth

class DepthDatasetLoader(Dataset):
    """"For loading RGB images and corresponding depth"""
    
    def __init__(self, dataset_directory):
        
        self.dataset_directory = dataset_directory
        self.rgb_images = sorted(glob.glob(self.dataset_directory + 'CameraRGB0/*'+'.png', recursive=True))
        self.drivedata = pd.read_csv('/home/shida/optical_flow_estimation/data0000/driving_data.csv')
        # test = drivedata.iloc[1]
        # print(1)
        #print(len(self.rgb_images))
        #self.matrix = []
        # for line in drivedata:
        #     line = line.rstrip()
        #     words = line.split(',')
            #self.matrix.append([float(words[],float(words[],float(words[])])
        #print(self.rgb_images)
        #self.depth_images = sorted(glob.glob(self.dataset_directory +"Depth" + '/**/*'+".png", recursive=True))
        
    def __len__(self):
        return 500
    
    def __getitem__(self, idx):
        rgb_img_file = self.rgb_images[idx]
        rannext = random.randint(1,5)
        curnum = int(rgb_img_file[-6:-4])
        if curnum==499:
            rannext = 0
        elif curnum>=494:
            rannext = 1
        #/home/shida/optical_flow_estimation/3dmotion/CameraRGB0/image_00000.png
        nextnum = int(rgb_img_file[-6:-4])+rannext
        if nextnum<10:
            tmp = '00'+str(nextnum)
        elif nextnum<100:
            tmp = '0'+str(nextnum)
        else:
            tmp = str(nextnum)
        rgb_img_file_2 = rgb_img_file[0:28] + tmp + '.png'
        #print(rgb_img_file_2)
        # depth_img_file = rgb_img_file[0:5] + 'DPT'+ rgb_img_file[8:15] + 'Dpt' + rgb_img_file[18::]
        # seg_img_file = rgb_img_file[0:5] + 'SEG' + rgb_img_file[8:15]+'Seg'+rgb_img_file[18::]
        #depth_img_file = self.depth_images[idx]
        #print(rgb_img_file[20::])
        #rgb_img = cv2.imread(rgb_img_file)[...,::-1] # bgr to rgb. Since opencv loads images as bgr


        rgb_img1 = read_rgb(rgb_img_file)
        rgb_img2 = read_rgb(rgb_img_file_2)
        # depth_img = read_depth(depth_img_file)    # Load as grayscale 
        # seg_img = read_rgb(seg_img_file)


        #depth_img = np.expand_dims(depth_img, 2)     # Add an extra channel dimension. Converts 
                                                     # (height, width) to (height, width, channel)
        # org_img_file = 'CameraGTRGB000'+rgb_img_file[20::]
        # org_img = cv2.imread(org_img_file)[...,::-1]
        # rgb_img = np.transpose(rgb_img, (2,0,1))     # Since Pytorch models take tensors 
        # org_img = np.transpose(org_img, (2,0,1)) 
        #print(rgb_img.type())
        #print(addLayer(rgb_img,dirct=rgb_img_file))
        #rgb_img = addLayer(rgb_img,dirct=rgb_img_file)
        #org_img = addLayer(org_img,dirct=rgb_img_file)
        #depth_img = np.transpose(depth_img, (2,0,1)) # in (channel, width, height) format
        data = dict()
        data['rgb1_file'] = rgb_img_file
        data['rgb2_file'] = rgb_img_file_2
        data['rgb1'] = rgb_img1.copy()
        data['rgb2'] = rgb_img2.copy()
        # data['depth'] = depth_img.copy()
        # data['seg'] = seg_img.copy()
        # print(self.drivedata.iloc[curnum]['pos_y'])
        # print(self.drivedata.iloc[nextnum]['pos_y'])
        x = self.drivedata.iloc[nextnum]['pos_x']-self.drivedata.iloc[curnum]['pos_x']
        y = self.drivedata.iloc[nextnum]['pos_y']-self.drivedata.iloc[curnum]['pos_y']
        z = self.drivedata.iloc[nextnum]['pos_z']-self.drivedata.iloc[curnum]['pos_z']
        data['t'] = np.array([x,y,z],dtype='f')
        # data['rgb_file'] = rgb_img_file
        # data['depth_file'] = depth_img_file
        # data['seg_file'] = seg_img_file
        return data

# dd_object = DepthDatasetLoader(dataset_directory = "data0000/")
# # print(dd_object[10].keys())
# #print(dd_object[10]['rgb1'].shape)
# print(dd_object[4]['y'])

# mydataloader = DataLoader(dd_object, batch_size=7, shuffle=True, num_workers=1, drop_last=True)

# for sub_iteration, minibatch in enumerate(mydataloader):
#     print(sub_iteration)
#     print(minibatch['rgb'].shape, minibatch['filenames'])