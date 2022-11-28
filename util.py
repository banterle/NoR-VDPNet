#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from PIL import Image
import numpy as np
import skimage.transform as st
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T
import cv2

#read an 8-bit image
def read_img(fname, grayscale=True):
    img = Image.open(fname)
    img = img.convert('L') if grayscale else img.convert('RGB')
    img_np = np.array(img);
    img_np = img_np.astype('float32')
    img_np /= 255.0
    img_torch = torch.FloatTensor(img_np)
    img_torch = img_torch.unsqueeze(0)
    #c = T.ToPILImage()
    #c(img_torch).save('test.png')
    return img_torch

#read a HDR image
def read_hdr(fname,  maxClip = 1e6, grayscale=True, log_range=True):
    img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    x = np.array(img, dtype=np.float)
    
    #no negative values
    x[x < 0.0] = 0.0
    #clipping
    x[x >= maxClip] = maxClip
    #remove NaNs and Infs
    x[np.isnan(x) == True] = maxClip
    x[np.isinf(x) == True] = maxClip
    
    if grayscale: #REC709 luminance
        x = 0.2126 * x[:,:,2] + 0.7152 * x[:,:,1] + 0.0722 * x[:,:,0]
        
    z = torch.FloatTensor(x)
    z = z.unsqueeze(0)
    
    if log_range:
        z = torch.log10(z + 1.0)
        
    return z

#read an 8-bit/32-bit image in MATLAB format
def read_mat(fname,  grayscale=True, log_range=True):
    x = loadmat(fname, verify_compressed_data_integrity=False)['image']
    
    if len(x.shape) == 3:
        x = 0.2126 * x[:,:,0] + 0.7152 * x[:,:,1] + 0.0722 * x[:,:,2]

    if log_range:  # perform log10(1 + image)
        x = np.log10(x + 1.0)

    x = x.astype('float32')
    return torch.FloatTensor(x)

#read an image
def load_image(fname, maxClip = 1e6, grayscale=True, log_range=True):
    filename, ext = os.path.splitext(fname)
    ext = ext.lower()
    
    if ext == '.mat':
       return read_mat(fname, grayscale, log_range)
    else:
        if (ext == '.exr') or (ext == '.hdr'):
            return read_hdr(fname, maxClip, grayscale, log_range)
        else:
            return read_img(fname, grayscale)

#data augmentation
def dataAugmentation_np(img, j):
    out = img.copy()
    jn = j % 7
    
    if (jn == 1):
        out = st.rotate(out, 90)
    elif (j == 2):
        out = st.rotate(out, 180)
    elif (jn == 3):
        out = st.rotate(out, 270)
    elif (jn == 4):
        out = np.fliplr(out)
        out = out.copy()
    elif (jn == 5):
        out = st.rotate(out, 90)
        out = np.fliplr(out)
        out = out.copy()
    elif (jn == 6):
        out = np.flipud(out)
        out = out.copy()
    return out
    
#
def torchDataAugmentation(img, j):
    img_out = []
    if(j == 0):
        img_out = img
    elif (j == 1):
        img_out = T.functional.rotate(img, 90)
    elif (j == 2):
        img_out = T.functional.rotate(img, 180)
    elif (j == 3):
        img_out = T.functional.rotate(img, 270)
    elif (j == 4):
        img_out = T.functional.hflip(img)
    elif (j == 5):
        img_tmp = T.functional.rotate(img, 90)
        img_out = T.functional.hflip(img_tmp)
        del img_tmp
    elif (j == 6):
        img_out = T.functional.vflip(img)
    elif (j == 7):
        img_out = T.functional.rotate(img, 30)
    elif (j == 8):
        img_out = T.functional.rotate(img, -30)
        
    return img_out

#plot a graph with train, validation, and test
def plotGraph(array_train, array_val, array_test, folder, suffix = "training"):
    fig = plt.figure(figsize=(10, 4))
    n = min([len(array_train), len(array_val), len(array_test)])
    plt.plot(np.arange(1, n + 1), array_train[0:n])# train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_val[0:n])  # val loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_test[0:n]) # test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation','test'], loc="upper left")
    title = os.path.join(folder, "plot_" + suffix + ".png")
    plt.savefig(title, dpi=600)
    plt.close(fig)
