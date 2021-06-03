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
from torchvision.transforms.functional import to_tensor
import numpy as np

def checkFile(base_dir, fn):
    stim_folder = os.path.join(base_dir, 'stim')
    full_name = os.path.join(stim_folder, fn)

    return os.path.isfile(full_name)

def split_data(data_dir, random_state=42, group=None, bPrecompGroup=True):

    data = os.path.join(data_dir, 'data.csv')
    data = pd.read_csv(data)
    
    img_fn = []
    q_val = []

    if group:
        print('Grouping')
        if bPrecompGroup == False:
           print('Groups transformations are online')
           n = len(data)

           for i in range(0, n):
               tmp0 = data.iloc[i].Distorted
               tmp1 = data.iloc[i].Q
               
               for j in range(0, group):
                   img_fn.append(tmp0)
                   q_val.append(tmp1)
           d = {'Distorted': img_fn, 'Q': q_val}
           data = pd.DataFrame(data=d)
        else:
           print('Groups are precomputed')
        data = [data[i:i + group] for i in range(0, len(data), group)]
    else:
        n = len(data)
        for i in range(0, n):
            tmp0 = data.iloc[i].Distorted
            tmp1 = data.iloc[i].Q
            
            if checkFile(data_dir, tmp0):
                img_fn.append(tmp0)
                q_val.append(tmp1)
            else:
                print(tmp0)
        d = {'Distorted': img_fn, 'Q': q_val}
        data = pd.DataFrame(data=d)
        
        print('No grouping')

    #split data into 80% train, 10% validation, and 10% test
    train, valtest = train_test_split(data, test_size=0.2, random_state=random_state)
    val, test = train_test_split(valtest, test_size=0.5, random_state=random_state)

    if group:
        train = pd.concat(train)
        val = pd.concat(val)
        test = pd.concat(test)

    return train, val, test

class HdrVdpDataset(Dataset):
    def __init__(self, data, base_dir, group = None, bPrecompGroup=True):
        self.data = data
        self.base_dir = base_dir
        self.group = group
        self.bPrecompGroup = bPrecompGroup

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        stim_folder = os.path.join(self.base_dir, 'stim')
        full_name = os.path.join(stim_folder, sample.Distorted)

        #print(full_name)
        stim = load_image(full_name)
        if self.group != None and (self.bPrecompGroup == False):
            stim = dataAugmentation_np(stim, index % self.group)

        stim = to_tensor(stim)
        q = torch.FloatTensor([sample.Q / 100.0])

        return stim, q

    def __len__(self):
        return len(self.data)
