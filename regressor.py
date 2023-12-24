#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn

#
#
#
class Regressor(nn.Module):

    #
    #
    #
    def __init__(self, in_size=1, out_size=1, params_size = None):
        super(Regressor, self).__init__()

        if params_size == None:
            params_size = 0

        self.params_size = params_size

        self.regressor = nn.Sequential(
                nn.Linear(in_size + params_size, 256),
                nn.ReLU(),
                nn.Linear(256, out_size),
                nn.Sigmoid()
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
        
        #if not self.training:
        #    q = q.clamp(0,1)
            
        return q

#
#
#
if __name__ == '__main__':
    model = Regressor()
    print(model)
