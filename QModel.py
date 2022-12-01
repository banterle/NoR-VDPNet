#
#Copyright (C) ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import re
import glob2
import torch

from model import QNet

class QModel:

    def __init__(self, run, grayscale = True):
        self.run = run
        
        if run.endswith('.pth'):
            ckpt = run
        else:
            ckpt_dir = os.path.join(run, 'ckpt')
            ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))
            assert ckpts, "No checkpoints to resume from!"

            def get_epoch(ckpt_url):
                s = re.findall("ckpt_e(\d+).pth", ckpt_url)
                epoch = int(s[0]) if s else -1
                return epoch, ckpt_url

            start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
            
        print('Checkpoint:', ckpt)
        
        ckpt = torch.load(ckpt)
        
        if grayscale:
            n_in =1
        else:
            n_in = 3
            
        model = QNet(n_in, 1)
        
        if torch.cuda.is_available():
           model = model.cuda()
        
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        self.model = model
    
    def getModel():
        return self.model
    
    def predict(self, stim):
        sz = stim.shape
        
        if len(sz) == 3:
            stim = stim.unsqueeze(0)
        
        with torch.no_grad():
             if torch.cuda.is_available():
                stim = stim.cuda()
        
             out = self.model(stim)
             out = out.data.cpu().numpy().squeeze()
        return out
