#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
#
class Regressor(nn.Module):

    #
    #
    #
    def __init__(self, in_size=1, out_size=1, params_size = None, bSigmoid = True):
        super(Regressor, self).__init__()

        if params_size == None:
            params_size = 0

        self.params_size = params_size

        if bSigmoid:
            self.regressor = nn.Sequential(
                    nn.Linear(in_size + params_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, out_size),
                    nn.Sigmoid()
                )
        else:
            self.regressor = nn.Sequential(
                    nn.Linear(in_size + params_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, out_size)
                )

    #
    #
    #
    def forward(self, features, params = None):
        if len(features.shape) == 4:
            features = features.mean(-1).mean(-1)
                
        if (self.params_size != 0) and (params != None):
            features = torch.cat((features, params), dim = 1)
            
        q = self.regressor(features)
        
        if not self.training:
            q = q.clamp(0,1)
            
        return q

#
#
#
class BlockQ(nn.Module):

    def __init__(self, in_size, out_size, std = 1):
        super(BlockQ, self).__init__()
    
        self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, 3, stride = std, padding=1),
                    nn.ReLU())

    def forward(self, input):
        return self.conv(input)

#
#
#
class QNet(nn.Module):

    def __init__(self, in_size=1, out_size=1, params_size = None):
        super(QNet, self).__init__()

        self.conv = nn.Sequential(
                    BlockQ(in_size, 32),
                    BlockQ(32, 32),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(32, 64),
                    BlockQ(64, 64),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(64, 128),
                    BlockQ(128, 128),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(128, 256),
                    BlockQ(256, 256),
                    nn.MaxPool2d(2),

                    BlockQ(256, 512),
                    BlockQ(512, 512, 2),
                    nn.MaxPool2d(2),
                    )
 
        self.regressor = Regressor(512, out_size, params_size)

    #
    #
    #
    def forward(self, stim, lmax = None):
        features = self.conv(stim)
        q = self.regressor(features, lmax)
        return q

if __name__ == '__main__':
    model = QNet()
    print(model)
