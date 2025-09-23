#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import argparse
from model import NoRVDPNetPPModel
from util import read_img_cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NoR-VDPNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, help='HDR_COMP (JPEG-XT compression), HDR_ITMO (inverse tone mapping), SDR (distortions for 8-bit images), and and SDR_TMO (tone mapping distortions).')
    parser.add_argument('img_folder', type=str, help='Base dir of run to evaluate')
    parser.add_argument('-dr', '--display_referred', type=str, default='yes', help='Do we need to apply the display? (yes/no)')
    parser.add_argument('-cs', '--colorspace', type=str, default='REC709', help='Color space of the input images')

    args = parser.parse_args()
        
    model = NoRVDPNetPPModel(args.mode, colorspace = args.colorspace, display_referred = args.display_referred)
        
    if (args.mode != 'SDR') and (args.mode != 'HDR_COMP') and (args.mode != 'HDR_ITMO') and (args.mode != 'SDR_TMO'):
        print('The mode ' + args.mode + ' selected is not supported.')
        print('Supported modes: HDR_ITMO, HDR_COMP, SDR, and SDR_TMO.')
        sys.exit()

    names_mat = [f for f in os.listdir(args.img_folder) if f.endswith('.mat')]
    names_hdr = [f for f in os.listdir(args.img_folder) if f.endswith('.hdr')]
    names_exr = [f for f in os.listdir(args.img_folder) if f.endswith('.exr')]
    names_hdr = sorted(names_mat + names_hdr + names_exr)

    names_jpg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpg')]
    names_jpeg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpeg')]
    names_png = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.png')]
    names_sdr = sorted(names_jpg + names_jpeg + names_png)
    
    names = names_hdr + names_sdr
    
    for name in names:
        fn = os.path.join(args.img_folder, name)
        p_model = float(model.predict(fn))
        print(name + " Q: " + str(round(p_model * 10000)/100))

    del model
