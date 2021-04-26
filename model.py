#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn

class QNet(nn.Module):

    def __init__(self, in_size=1, out_size=1):
        super(QNet, self).__init__()

        pad = 1
        std = 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=pad),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=std, padding=pad),
            nn.ReLU(),
            #####################################################
            nn.Conv2d(32, 64, 3, padding=pad),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=std, padding=pad),
            nn.ReLU(),
            #####################################################
            nn.Conv2d(64, 128, 3, padding=pad),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=std, padding=pad),
            nn.ReLU(),
            #####################################################
            nn.Conv2d(128, 256, 3, padding=pad),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=std, padding=pad),
            nn.ReLU(),
            #####################################################
            nn.Conv2d(256, 512, 3, padding=pad),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=std, padding=pad),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_size),
        )

    def forward(self, stim):
        features = self.conv(stim)
        features_pooled = features.mean(-1).mean(-1)
        q = self.regressor(features_pooled)
        
        if not self.training:
            q = q.clamp(0,1)
            
        return q

if __name__ == '__main__':
    model = QNet()
    print(model)
