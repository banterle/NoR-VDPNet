#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor
from scipy.io import loadmat
from PIL import Image
import numpy as np

#read an 8-bit image
def read_img(fname, grayscale=True):
    img = Image.open(fname)
    img = img.convert('L') if grayscale else img.convert('RGB')
    x = to_tensor(img)
    return x

#read an 8-bit/32-bit image in MATLAB format
def read_mat(fname,  grayscale=True, log_range=True):
    x = loadmat(fname, verify_compressed_data_integrity=False)['image']
    x = torch.FloatTensor(x)
    
    if (x.ndimension() == 3) and grayscale:
        x = x.transpose(2, 0)
        x = torch.sum(x, dim = 0) / 3
        x = x.unsqueeze(0)
    
    if log_range:  # perform log10(1 + image)
        x += 1
        torch.log10(x, out = x)
    elif x.ndimension() == 2:
        x = x.unsqueeze(0)
    return x

#read an image
def load_image(fname,  grayscale=True, log_range=True):
    filename, ext = os.path.splitext(fname)
    if ext == '.mat':
       return read_mat(fname, grayscale, log_range)
    else:
       return read_img(fname, grayscale)

#plot a graph with train, validation, and test
def plotGraph(array_train, array_val, array_test, folder):
    fig = plt.figure(figsize=(10, 4))
    n = min([len(array_train), len(array_val), len(array_test)])
    plt.plot(np.arange(1, n + 1), array_train[0:n])# train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_val[0:n])  # val loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_test[0:n]) # test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation','test'], loc="upper left")
    title = os.path.join(folder, "plot.png")
    plt.savefig(title, dpi=600)
    plt.close(fig)
