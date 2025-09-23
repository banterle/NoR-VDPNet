#
#Copyright (C) ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import re
import glob2
import torch

from model import QNet

#
#
#
class QModel:

    #
    #
    #
    def __init__(self, run, btype = 2, maxClip = 1400, grayscale = True, colorspace = 'REC709', display_referred = 'yes'):
        url_str = 'http://www.banterle.com/francesco/projects/norvdpnet/'
        
        bDone = False
        
        args_mode = run
        if args_mode == 'SDR':
            try:
                model = self.setup_aux('weights/norvdpnet_sdr.pth', maxClip, btype, grayscale, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'norvdpnet_sdr.pth', maxClip, btype, grayscale, colorspace, display_referred)
            bDone = True

        if args_mode == 'HDR_COMP':
            try:
                model = self.setup_aux('weights/norvdpnet_hdrc.pth', maxClip, btype, grayscale, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'norvdpnet_hdrc.pth', maxClip, btype, grayscale, colorspace, display_referred)
            bDone = True

        if args_mode == 'HDR_ITMO':
            try:
                model = self.setup_aux('weights/norvdpnet_itmo.pth', maxClip, btype, grayscale, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'norvdpnet_itmo.pth', maxClip, btype, grayscale, colorspace, display_referred)
            bDone = True

        if args_mode == 'SDR_TMO':
            try:
                model = self.setup_aux('weights/norvdpnet_tmo.pth', maxClip, btype, grayscale, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'norvdpnet_tmo.pth', maxClip, btype, grayscale, colorspace, display_referred)
            bDone = True
            
        if bDone == False:
            self.setup_aux(run, btype, maxClip, grayscale, colorspace, display_referred)
    
    #
    #
    #
    def setup_aux(self, run, btype = 2, maxClip = 1400, grayscale = True, colorspace = 'REC709', display_referred = 'yes'):
        self.run = run
        ext = os.path.splitext(run)[1]
        
        if ext == '':
            ckpt_dir = os.path.join(run, 'ckpt')
            ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))
            assert ckpts, "No checkpoints to resume from!"

            def get_epoch(ckpt_url):
                s = re.findall("ckpt_e(\\d+).pth", ckpt_url)
                epoch = int(s[0]) if s else -1
                return epoch, ckpt_url

            start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
            print('Checkpoint:', ckpt)
        else:
            if 'http://' in run:
                cache_dir = os.path.expanduser('./weights')
                os.makedirs(cache_dir, exist_ok=True)

                filename = os.path.basename(run)

                cached_path = os.path.join(cache_dir, filename)

                if not os.path.exists(cached_path):
                    urllib.request.urlretrieve(run, cached_path)

                ckpt = cached_path

            else:
                ckpt = run

        bLoad = True

        if ckpt == 'none.pth':
            bLoad = False
            
        if grayscale:
            n_in =1
        else:
            n_in = 3
                        
        if bLoad:
            if torch.cuda.is_available():
                ckpt = torch.load(ckpt, weights_only=True)
            else:
                ckpt = torch.load(ckpt, weights_only=True, map_location=torch.device('cpu'))

        if 'grayscale' in ckpt:
            if ckpt['grayscale']:
                n_in =1
            else:
                n_in = 3
                        
        model = QNet(n_in, 1)

        model.load_state_dict(ckpt['model'])

        if(torch.cuda.is_available()):
            model = model.cuda()

        model.eval()
        
        self.model = model

        self.colorspace = colorspace
        self.maxClip = maxClip
        self.grayscale = grayscale
        self.display_referred = (display_referred == 'yes')
    
    #
    #
    #
    def getModel(self):
        return self.model

    #
    #
    #
    def predict(self, fn):
        stim = read_img_cv2(fn, maxClip = self.maxClip, grayscale = self.grayscale, colorspace = self.colorspace, display_referred = self.display_referred)
        stim = stim.unsqueeze(0)

        if torch.cuda.is_available():
            stim = stim.cuda()

        with torch.no_grad():
            out = self.model(stim)

        out = out.data.cpu().numpy().squeeze()

        return out
