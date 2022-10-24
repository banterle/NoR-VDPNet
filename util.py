#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from PIL import Image
import numpy as np
import skimage.transform as st
from torchvision.transforms.functional import to_tensor

import cv2

#read an 8-bit image
def read_img(fname, grayscale=True):
    img = Image.open(fname)
    img = img.convert('L') if grayscale else img.convert('RGB')
    img_np = np.array(img);
    img_np = img_np.astype('float32')
    img_np /= 255.0
    return torch.FloatTensor(img_np)

#read a HDR image
def read_hdr(fname,  grayscale=True, log_range=True):
    img = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    x = np.array(img, dtype=np.float)
    if grayscale: #luminance
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
def load_image(fname,  grayscale=True, log_range=True):
    filename, ext = os.path.splitext(fname)
    ext = ext.lower()
    
    if ext == '.mat':
       return read_mat(fname, grayscale, log_range)
    else:
        if (ext == '.exr') or (ext == '.hdr'):
            return read_hdr(fname, grayscale, log_range)
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
