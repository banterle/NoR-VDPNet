#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import pandas as pd
import torch
from util import load_image, dataAugmentation_np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#
#
#
def checkFile(base_dir, fn):
    full_name = os.path.join(base_dir, fn)
    return os.path.isfile(full_name)

#
#
#
def read_data_split(data_dir, group=None, bPrecompGroup=True):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train.sort_values(by=['Distorted'], inplace=True)

    val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    val.sort_values(by=['Distorted'], inplace=True)

    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test.sort_values(by=['Distorted'], inplace=True)

    return train, val, test
    
#
#
#
def filterLuminance(data, group = None, thr = 500):
    if group == None:
        group = 1
        
    img_fn = []
    q_val = []
    L_max = []
    bCheck = thr > 0.0

    n = len(data)
    for i in range(0, n):
        if bCheck:
            if np.abs(data.iloc[i].Lmax - thr) < 1.0:
                tmp0 = data.iloc[i].Distorted
                tmp1 = data.iloc[i].Q
                tmp2 = data.iloc[i].Lmax
        
                for j in range(0, group):
                    img_fn.append(tmp0)
                    q_val.append(tmp1)
                    L_max.append(tmp2)
        else:
            tmp0 = data.iloc[i].Distorted
            tmp1 = data.iloc[i].Q
            tmp2 = data.iloc[i].Lmax
        
            for j in range(0, group):
                img_fn.append(tmp0)
                q_val.append(tmp1)
                L_max.append(tmp2)

    d = {'Distorted': img_fn, 'Lmax': L_max, 'Q': q_val}
    data = pd.DataFrame(data=d)
    
    return data, q_val
    
    
def filterLuminanceSimple(data, group = None):
    if group == None:
        group = 1
        
    img_fn = []
    q_val = []

    n = len(data)
    for i in range(0, n):
        tmp0 = data.iloc[i].Distorted
        tmp1 = data.iloc[i].Q
        
        for j in range(0, group):
            img_fn.append(tmp0)
            q_val.append(tmp1)

    d = {'Distorted': img_fn, 'Q': q_val}
    data = pd.DataFrame(data=d)
    
    return data, q_val

#
#
#
def split_data(data_dir, random_state=42, group=None, bPrecompGroup=True, thr = -1):

    data = os.path.join(data_dir, 'data.csv')
    data = pd.read_csv(data)
    
    if group:
        print('Grouping')
        if bPrecompGroup == False:
            print('Groups transformations are online')
            try:
                data, q_val = filterLuminance(data, group, thr)
            except:
                data, q_val = filterLuminanceSimple(data, group)
        else:
            print('Groups are precomputed')
            try:
                data, q_val = filterLuminance(data, None, thr)
            except:
                data, q_val = filterLuminanceSimple(data, group)

        data = [data[i:i + group] for i in range(0, len(data), group)]
    else:
        print('No grouping')
        try:
            data, q_val = filterLuminance(data, None, thr)
        except:
            data, q_val = filterLuminanceSimple(data, group)

    plt.clf()
    sns.distplot(q_val, kde=True, rug=True, bins=100)
    plt.savefig('hist_q.png')

    #split data into 80% train, 10% validation, and 10% test
    train, valtest = train_test_split(data, test_size=0.2, random_state=random_state)
    val, test = train_test_split(valtest, test_size=0.5, random_state=random_state)
    
    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)

    return train, val, test

class HdrVdpDataset(Dataset):
    def __init__(self, data, base_dir, group = None, bPrecompGroup=True, bScaling = False, colorspace = 'REC709', color = 'gray'):
        self.data = data
        self.base_dir = base_dir
        self.group = group
        self.bPrecompGroup = bPrecompGroup
        self.bScaling = bScaling
        self.colorspace = colorspace
        
        self.bGrayscale = (color == 'gray')
        
        if self.bScaling:
            print('Scaling is active')
        else:
            print('Scaling is disabled')

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        full_name = os.path.join(self.base_dir, sample.Distorted)

        #print(full_name)
        stim = load_image(full_name, grayscale = self.bGrayscale, colorspace = self.colorspace)
        
        if self.group != None and (self.bPrecompGroup == False):
            stim = dataAugmentation_np(stim, index % self.group)
        
        if self.bScaling:
            q = torch.FloatTensor([sample.Q / 100.0])
        else:
            q = torch.FloatTensor([sample.Q])

        return stim, q

    def __len__(self):
        return len(self.data)
