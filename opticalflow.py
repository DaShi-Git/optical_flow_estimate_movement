import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import copy
import glob
import cv2
FOLDER = "/home/shida/optical_flow_estimation/"

def read_rgb(rgb_file):
    rgb = io.imread(rgb_file)
    return rgb

def read_depth(depth_file):
    depth = io.imread(depth_file)
#     Reference: https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256)
    depth = depth * (1/ (256 * 256 * 256 - 1))
    return depth


# def gunner(first, second):
def gunner(first,second):
    # cam_folder = os.path.join(FOLDER, "CameraRGB0")
    # all_images = sorted(glob.glob(cam_folder + '/**/*.png', recursive=True))
    # first = all_images[0]
    # second = all_images[1]
    # num_images = len(all_images)
    prvs = read_rgb('/home/shida/optical_flow_estimation/'+first)
    #prvs = first
    hsv = np.zeros_like(prvs)
    hsv[...,1] = 255
    prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)

    #for i in range(1):
    frame2 = read_rgb('/home/shida/optical_flow_estimation/'+ second)
    #frame2 = second
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None,pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=10, poly_n=5, poly_sigma=1.2,
                                    flags=0)
                                        
                                        
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #cv2.imshow('frame2',np.concatenate((frame2[...,::-1], rgb), axis=1))
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
        #cv2.imwrite('opticalfb.png',frame2)
    #cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

    #cv2.destroyAllWindows()
    return rgb
#gunner()